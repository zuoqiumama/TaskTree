import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import Bottleneck, ResNet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection import MaskRCNN
from collections import OrderedDict
from models.custom_modules import ResCBAM, SCConv, ProtoNet, CrossScaleAttention

class CustomBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, 
                 use_cbam=False, use_scconv=False):
        super(CustomBottleneck, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        self.use_cbam = use_cbam
        self.use_scconv = use_scconv
        
        # SCConv replacement for conv2 (3x3)
        if self.use_scconv:
            width = int(planes * (base_width / 64.)) * groups
            self.conv2 = SCConv(width, group_num=groups, stride=stride)

        if self.use_cbam:
            self.cbam = ResCBAM(planes * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.use_cbam:
            out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class CustomResNet(ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, use_cbam=False, use_scconv=False):
        # Initialize parent with standard args, we will overwrite layers
        super(CustomResNet, self).__init__(block, layers, num_classes, zero_init_residual,
                                           groups, width_per_group, replace_stride_with_dilation,
                                           norm_layer)
        
        self.use_cbam = use_cbam
        self.use_scconv = use_scconv
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        
        # Re-create layers with custom flags
        self.inplanes = 64 # Reset inplanes
        self.dilation = 1 # Reset dilation
        self.layer1 = self._make_layer(block, 64, layers[0], use_cbam=use_cbam, use_scconv=use_scconv)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], use_cbam=use_cbam, use_scconv=use_scconv)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], use_cbam=use_cbam, use_scconv=use_scconv)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], use_cbam=use_cbam, use_scconv=use_scconv)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, use_cbam=False, use_scconv=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                torchvision.models.resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, 
                            use_cbam=use_cbam, use_scconv=use_scconv))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                use_cbam=use_cbam, use_scconv=use_scconv))

        return nn.Sequential(*layers)

class CustomMaskRCNN(MaskRCNN):
    def __init__(self, backbone, num_classes=None, 
                 use_proto=False, use_csa=False,
                 **kwargs):
        super(CustomMaskRCNN, self).__init__(backbone, num_classes, **kwargs)
        self.use_proto = use_proto
        self.use_csa = use_csa
        
        if self.use_csa:
            out_channels = backbone.out_channels
            # FPN outputs 256 channels
            self.csa = CrossScaleAttention([out_channels]*4, out_channels)
            
        if self.use_proto:
            self.proto_net = ProtoNet(backbone.out_channels, num_classes)

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
            
        refined_feat = None
        if self.use_csa:
            # Only use the first 4 features (P2, P3, P4, P5) which correspond to keys '0', '1', '2', '3'
            # 'pool' key might be present but we don't use it for CSA
            csa_keys = ['0', '1', '2', '3']
            feat_list = [features[k] for k in csa_keys if k in features]
            refined_feat = self.csa(feat_list)

        losses = {}
        
        if self.use_proto:
            if refined_feat is not None:
                 proto_input = refined_feat
            else:
                 proto_input = list(features.values())[-1]
            
            proto_logits = self.proto_net(proto_input)
            
            if self.training:
                gt_labels = [t['labels'] for t in targets]
                device = proto_logits.device
                batch_size = len(gt_labels)
                target_one_hot = torch.zeros(batch_size, self.proto_net.num_classes, device=device)
                for i, labels in enumerate(gt_labels):
                    if len(labels) > 0:
                        target_one_hot[i, labels] = 1.0
                
                proto_loss = nn.functional.binary_cross_entropy_with_logits(proto_logits, target_one_hot)
                losses['loss_proto'] = proto_loss

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        
        if self.training:
            losses.update(proposal_losses)
            losses.update(detector_losses)
            return losses
        
        return detections

def get_custom_maskrcnn(num_classes, use_cbam=False, use_scconv=False, use_proto=False, use_csa=False, pretrained_backbone=True, **kwargs):
    backbone = CustomResNet(CustomBottleneck, [3, 4, 6, 3], use_cbam=use_cbam, use_scconv=use_scconv)
    
    if pretrained_backbone:
        state_dict = torchvision.models.resnet50(pretrained=True).state_dict()
        model_dict = backbone.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        backbone.load_state_dict(model_dict)
        
    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
    in_channels_list = [256, 512, 1024, 2048]
    out_channels = 256
    backbone_with_fpn = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
    
    model = CustomMaskRCNN(backbone_with_fpn, num_classes=num_classes, use_proto=use_proto, use_csa=use_csa, **kwargs)
    return model
