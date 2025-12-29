import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import Bottleneck, ResNet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models.detection import MaskRCNN
from collections import OrderedDict
from models.custom_modules import ResCBAM, SCConv, ProtoNet, CrossScaleAttention

class BackboneWithPAFPN(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels, extra_blocks=None, use_csa=False):
        super(BackboneWithPAFPN, self).__init__()
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.return_layers = return_layers
        
        # Standard FPN (Top-Down)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=None, 
        )
        self.extra_blocks = extra_blocks
        
        # PAFPN (Bottom-Up)
        self.downsample_convs = nn.ModuleDict()
        self.downsample_convs['0_1'] = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.downsample_convs['1_2'] = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.downsample_convs['2_3'] = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        
        self.use_csa = use_csa
        if self.use_csa:
            # CSA for N3, N4, N5 calculation
            self.csa_n3 = CrossScaleAttention([out_channels, out_channels], out_channels)
            self.csa_n4 = CrossScaleAttention([out_channels, out_channels], out_channels)
            self.csa_n5 = CrossScaleAttention([out_channels, out_channels], out_channels)
            
            # Learnable weights for CSA fusion
            self.w_n3 = nn.Parameter(torch.tensor(0.5))
            self.w_n4 = nn.Parameter(torch.tensor(0.5))
            self.w_n5 = nn.Parameter(torch.tensor(0.5))
            
        self.out_channels = out_channels
        self.pafpn_convs = nn.ModuleDict()
        self.pafpn_convs['n2'] = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.pafpn_convs['n3'] = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.pafpn_convs['n4'] = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.pafpn_convs['n5'] = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.body(x)
        
        # FPN
        fpn_out = self.fpn(x)
        
        # PAFPN
        n2 = fpn_out['0'] # P2
        p3 = fpn_out['1']
        p4 = fpn_out['2']
        p5 = fpn_out['3']
        n2 = self.pafpn_convs['n2'](n2)
        # N3
        n2_down = self.downsample_convs['0_1'](n2)
        if self.use_csa:
            n3 = self.csa_n3([p3, n2_down]) * self.w_n3 + p3
        else:
            n3 = n2_down + p3
        n3 = self.pafpn_convs['n3'](n3)

        # N4
        n3_down = self.downsample_convs['1_2'](n3)
        if self.use_csa:
            n4 = self.csa_n4([p4, n3_down]) * self.w_n4 + p4
        else:
            n4 = n3_down + p4
        n4 = self.pafpn_convs['n4'](n4)
        
        # N5
        n4_down = self.downsample_convs['2_3'](n4)
        if self.use_csa:
            n5 = self.csa_n5([p5, n4_down]) * self.w_n5 + p5
        else:
            n5 = n4_down + p5
        n5 = self.pafpn_convs['n5'](n5)
            
        results = OrderedDict()
        results['0'] = n2
        results['1'] = n3
        results['2'] = n4
        results['3'] = n5
        
        if self.extra_blocks is not None:
            results_list = list(results.values())
            names = list(results.keys())
            results_list, names = self.extra_blocks(results_list, x, names)
            results = OrderedDict([(k, v) for k, v in zip(names, results_list)])
            
        return results

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
        
        if self.use_cbam:
            # Apply ResCBAM before Layer 1 (C2)
            # The input to layer1 is 64 channels (from stem)
            self.cbam_stem = ResCBAM(64)
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        
        # Re-create layers with custom flags
        self.inplanes = 64 # Reset inplanes
        self.dilation = 1 # Reset dilation
        # Note: We pass use_cbam=False to blocks because we only use it once at stem
        self.layer1 = self._make_layer(block, 64, layers[0], use_cbam=False, use_scconv=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], use_cbam=False, use_scconv=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], use_cbam=False, use_scconv=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], use_cbam=False, use_scconv=False)

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
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.use_cbam:
            x = self.cbam_stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.avgpool:
             x = self.avgpool(x)
             x = torch.flatten(x, 1)
             x = self.fc(x)

        return x

class CustomMaskRCNN(MaskRCNN):
    def __init__(self, backbone, num_classes=None, 
                 use_proto=False,
                 **kwargs):
        super(CustomMaskRCNN, self).__init__(backbone, num_classes, **kwargs)
        self.use_proto = use_proto
        
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
            
        losses = {}
        
        if self.use_proto:
            # Use the last feature map (N5) for ProtoNet
            proto_input = list(features.values())[-1]
            
            proto_logits = self.proto_net(proto_input)
            
            if self.training:
                gt_labels = [t['labels'] for t in targets]
                device = proto_logits.device
                batch_size = len(gt_labels)
                target_one_hot = torch.zeros(batch_size, self.proto_net.num_classes, device=device)
                for i, labels in enumerate(gt_labels):
                    if len(labels) > 0:
                        # Safety check for labels
                        valid_mask = (labels >= 0) & (labels < self.proto_net.num_classes)
                        if not valid_mask.all():
                            invalid_labels = labels[~valid_mask]
                            print(f"WARNING: Found invalid labels: {invalid_labels} for num_classes {self.proto_net.num_classes}")
                            labels = labels[valid_mask]
                        
                        if len(labels) > 0:
                            target_one_hot[i, labels] = 1.0
                
                if torch.isnan(proto_logits).any() or torch.isinf(proto_logits).any():
                     print("WARNING: proto_logits contains NaN or Inf")

                proto_loss = nn.functional.binary_cross_entropy_with_logits(proto_logits, target_one_hot)
                losses['loss_proto'] = proto_loss

        # Debugging: Check types of features
        # for k, v in features.items():
        #     print(f"Feature {k}: type={type(v)}")
        #     if isinstance(v, list):
        #         print(f"Feature {k} is a list! Length: {len(v)}")
        #         if len(v) > 0:
        #             print(f"Element 0 type: {type(v[0])}")

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        
        if self.training:
            losses.update(proposal_losses)
            losses.update(detector_losses)
            return losses
        
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        return detections

def get_custom_maskrcnn(num_classes, use_cbam=False, use_scconv=False, use_proto=False, use_csa=False, pretrained_backbone=True, **kwargs):
    # Pass use_cbam to CustomResNet, where it will be applied once before Layer 1
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
    # We don't need use_cbam here anymore as it's handled in the backbone
    backbone_with_fpn = BackboneWithPAFPN(backbone, return_layers, in_channels_list, out_channels, use_csa=use_csa)
    
    model = CustomMaskRCNN(backbone_with_fpn, num_classes=num_classes, use_proto=use_proto, **kwargs)
    return model
