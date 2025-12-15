"""
YOLO分割模型封装器
用于在DISCO智能体中替换Mask R-CNN
提供与原有接口兼容的预测功能
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class YOLOSegWrapper(nn.Module):
    """
    YOLO分割模型封装器
    提供与Mask R-CNN类似的接口，便于在智能体中无缝替换
    """
    def __init__(self, model_path, num_classes=85, conf_thresh=0.5, iou_thresh=0.5, device='cuda:0'):
        super().__init__()
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Please run: pip install ultralytics")
        
        self.model = YOLO(model_path)
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        
    def to(self, device):
        """移动模型到指定设备"""
        self.device = device
        return self
    
    def eval(self):
        """设置为评估模式"""
        self.model.model.eval()
        return self
    
    def train(self, mode=True):
        """设置为训练模式"""
        self.model.model.train(mode)
        return self
    
    def forward(self, images, targets=None):
        """
        前向传播
        
        Args:
            images: list of tensors [C, H, W] 或 batch tensor [B, C, H, W]
                   值范围 [0, 1]
            targets: 训练时的目标，推理时为None
        
        Returns:
            推理模式: list of dict, 每个dict包含 'boxes', 'labels', 'scores', 'masks'
            训练模式: dict of losses
        """
        if targets is not None:
            # 训练模式 - YOLO使用自己的训练循环
            raise NotImplementedError("Use ultralytics native training instead")
        
        # 推理模式
        return self.predict(images)
    
    def predict(self, images):
        """
        预测函数
        
        Args:
            images: list of tensors [C, H, W] 或单个RGB numpy数组 [H, W, C]
        
        Returns:
            list of dict, 每个dict包含:
                - boxes: tensor [N, 4] (x1, y1, x2, y2)
                - labels: tensor [N] 类别ID (从1开始，与MRCNN一致)
                - scores: tensor [N] 置信度
                - masks: tensor [N, 1, H, W] 二值mask
        """
        # 准备输入
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:  # [B, C, H, W]
                images = [img for img in images]
            else:  # [C, H, W]
                images = [images]
        
        input_images = []
        original_sizes = []
        
        for img in images:
            if isinstance(img, torch.Tensor):
                # tensor [C, H, W] -> numpy [H, W, C] BGR
                img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            elif isinstance(img, np.ndarray):
                if img.shape[0] == 3:  # [C, H, W]
                    img_np = (np.transpose(img, (1, 2, 0)) * 255).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                else:  # [H, W, C]
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[-1] == 3 else img
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            
            input_images.append(img_bgr)
            original_sizes.append(img_bgr.shape[:2])
        
        # YOLO推理
        results = self.model.predict(
            input_images,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False,
            retina_masks=True,
        )
        
        # 转换输出格式
        outputs = []
        for result, orig_size in zip(results, original_sizes):
            output = {}
            
            if result.masks is not None and len(result.masks) > 0:
                # 边界框
                boxes = result.boxes.xyxy.cpu()  # [N, 4]
                
                # 类别 (YOLO从0开始，转换为从1开始以匹配MRCNN)
                labels = result.boxes.cls.cpu().int() + 1
                
                # 置信度
                scores = result.boxes.conf.cpu()
                
                # Masks [N, H, W] -> [N, 1, H, W]
                masks = result.masks.data.cpu()
                masks = masks.unsqueeze(1)  # [N, 1, H, W]
                
                output['boxes'] = boxes
                output['labels'] = labels
                output['scores'] = scores
                output['masks'] = masks
            else:
                # 无检测结果
                output['boxes'] = torch.zeros((0, 4))
                output['labels'] = torch.zeros((0,), dtype=torch.int64)
                output['scores'] = torch.zeros((0,))
                output['masks'] = torch.zeros((0, 1, orig_size[0], orig_size[1]))
            
            outputs.append(output)
        
        return outputs
    
    def predict_single(self, image, return_numpy=True):
        """
        单张图像预测（便捷接口）
        
        Args:
            image: numpy数组 [H, W, C] RGB 或 tensor [C, H, W]
            return_numpy: 是否返回numpy数组
        
        Returns:
            dict with 'boxes', 'labels', 'scores', 'masks'
        """
        outputs = self.predict([image])[0]
        
        if return_numpy:
            return {
                'boxes': outputs['boxes'].numpy(),
                'labels': outputs['labels'].numpy(),
                'scores': outputs['scores'].numpy(),
                'masks': outputs['masks'].numpy(),
            }
        return outputs
    
    def get_segmentation_mask(self, image, target_class=None, threshold=0.5):
        """
        获取特定类别的分割mask
        
        Args:
            image: 输入图像
            target_class: 目标类别ID (从1开始)，None表示返回所有类别
            threshold: mask阈值
        
        Returns:
            numpy数组 [H, W] 或 [N, H, W]
        """
        output = self.predict_single(image, return_numpy=True)
        
        if len(output['labels']) == 0:
            h, w = image.shape[:2] if isinstance(image, np.ndarray) else image.shape[1:3]
            return np.zeros((h, w), dtype=np.uint8)
        
        masks = output['masks'][:, 0]  # [N, H, W]
        labels = output['labels']
        scores = output['scores']
        
        if target_class is not None:
            # 筛选特定类别
            mask_idx = np.where((labels == target_class) & (scores >= threshold))[0]
            if len(mask_idx) == 0:
                return np.zeros(masks.shape[1:], dtype=np.uint8)
            # 合并同类别的mask
            selected_masks = masks[mask_idx]
            combined_mask = np.max(selected_masks, axis=0)
            return (combined_mask > 0.5).astype(np.uint8)
        else:
            # 返回所有mask
            return (masks > 0.5).astype(np.uint8)


def load_yolo_seg_model(model_path, device='cuda:0', conf_thresh=0.5):
    """
    加载YOLO分割模型的便捷函数
    
    Args:
        model_path: 模型权重路径
        device: 设备
        conf_thresh: 置信度阈值
    
    Returns:
        YOLOSegWrapper实例
    """
    return YOLOSegWrapper(model_path, conf_thresh=conf_thresh, device=device)


if __name__ == '__main__':
    """测试模型封装器"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--image', type=str, required=True, help='Test image path')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    
    # 加载模型
    model = load_yolo_seg_model(args.model, args.device)
    model.eval()
    
    # 读取测试图像
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 预测
    output = model.predict_single(image)
    
    print(f"Detected {len(output['labels'])} objects")
    for i, (label, score) in enumerate(zip(output['labels'], output['scores'])):
        print(f"  {i}: class={label}, score={score:.3f}")
