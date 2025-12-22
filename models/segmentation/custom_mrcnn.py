import torch
import torch.nn as nn
import torchvision
from models.segmentation.yolo.custom_modules import (
    CBAM, ResCBAM, MinResCBAM, IMinCBAM, 
    ProtoNet, CrossScaleAttention, SCConv
)

CUSTOM_MODULES = {
    'CBAM': CBAM,
    'ResCBAM': ResCBAM,
    'MinResCBAM': MinResCBAM,
    'IMinCBAM': IMinCBAM,
    'ProtoNet': ProtoNet,
    'CrossScaleAttention': CrossScaleAttention,
    'SCConv': SCConv
}

def get_custom_mrcnn(num_classes, custom_module_name=None, pretrained=True, **kwargs):
    """
    Builds a Mask R-CNN model with optional custom modules inserted into the backbone.
    
    Args:
        num_classes (int): Number of classes (including background).
        custom_module_name (str): Name of the custom module to insert.
        pretrained (bool): Whether to use pretrained backbone weights.
    """
    # Load the standard model
    # We start with pretrained=pretrained to get the backbone weights if requested
    # But we might need to reset the head if num_classes is different
    
    # Note: torchvision.models.detection.maskrcnn_resnet50_fpn handles loading weights
    # If we want to modify the backbone, we should do it after loading.
    
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=pretrained, 
        box_score_thresh=0.5,
        **kwargs
    )
    
    # Replace the head to match our number of classes
    # (This logic was in train_mrcnn.py, moving it here for consistency)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    if custom_module_name:
        if custom_module_name not in CUSTOM_MODULES:
            raise ValueError(f"Custom module {custom_module_name} not found. Available: {list(CUSTOM_MODULES.keys())}")
        
        module_class = CUSTOM_MODULES[custom_module_name]
        print(f"Inserting {custom_module_name} into backbone layers...")
        
        # The backbone body is a ResNet
        # We will insert the module after each layer (layer1, layer2, layer3, layer4)
        backbone_body = model.backbone.body
        
        # Layer channels for ResNet50
        layer_channels = {
            'layer1': 256,
            'layer2': 512,
            'layer3': 1024,
            'layer4': 2048
        }
        
        for layer_name, channels in layer_channels.items():
            if hasattr(backbone_body, layer_name):
                old_layer = getattr(backbone_body, layer_name)
                # Create new sequential with custom module appended
                # Note: Some custom modules might need extra args. 
                # We assume default args or c1=channels are sufficient.
                
                # Check if module needs c2 or other args
                # Most defined in custom_modules take c1 as first arg.
                
                new_module = module_class(channels)
                
                new_layer = nn.Sequential(*list(old_layer.children()), new_module)
                setattr(backbone_body, layer_name, new_layer)
                print(f"  Added {custom_module_name} to {layer_name} (channels={channels})")
                
    return model
