"""
YOLO实例分割模型评估脚本
使用与evaluate_mrcnn.py相同的评估指标，便于模型对比
"""
import sys
import os
# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

import cv2
import numpy as np
import torch
from datasets.segmentation_dataset import build_seg_dataloader
from utils import utils, arguments
from utils.alfred_interest_objects import ALFRED_INTEREST_OBJECTS
import logging
from tqdm import tqdm

# 检查是否安装了ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Please run: pip install ultralytics")

torch.multiprocessing.set_sharing_strategy('file_system')


def resize_mask_to_target(mask, target_size):
    """将mask resize到目标尺寸"""
    if mask.shape[-2:] == target_size:
        return mask
    from PIL import Image
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_resized = mask_pil.resize((target_size[1], target_size[0]), Image.NEAREST)
    return (np.array(mask_resized) > 127).astype(np.float32)


def evaluate_yolo_on_alfred(model, args, logger, splits=['valid_seen', 'valid_unseen']):
    """
    使用与MRCNN相同的数据加载器和指标评估YOLO模型
    """
    device = args.gpu[0] if isinstance(args.gpu, list) else args.gpu
    
    # 使用与MRCNN相同的数据集设置
    args.image_size = 320
    args.mask_size = 300
    
    results_all = {}
    
    for split in splits:
        dataloader = build_seg_dataloader('AlfredSegImageDataset', split=split, args=args)
        logger.info(f'=================== {split} ({len(dataloader.dataset)} images) ==================')
        
        all_pred, all_gt = [], []
        
        for i, blobs in enumerate(tqdm(dataloader, desc=f'Evaluating {split}')):
            images, targets = blobs
            if len(images) == 0:
                continue
            
            # 准备输入图像（YOLO需要BGR格式的numpy数组或tensor）
            input_images = []
            for img in images:
                # img是 [C, H, W] float tensor, 范围[0,1]
                img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                input_images.append(img_bgr)
            
            # YOLO推理
            with torch.no_grad():
                results = model.predict(
                    input_images,
                    imgsz=args.image_size,
                    conf=0.5,
                    iou=0.5,
                    device=device,
                    verbose=False,
                    retina_masks=True,  # 高质量mask
                )
            
            # 收集GT
            for tgt in targets:
                all_gt.append({
                    'label': tgt['labels'].numpy(),
                    'mask': tgt['masks'].numpy()
                })
            
            # 收集预测结果
            for result in results:
                if result.masks is not None and len(result.masks) > 0:
                    # YOLO masks: [N, H, W]
                    masks_np = result.masks.data.cpu().numpy()
                    # YOLO类别从0开始，MRCNN从1开始，需要+1
                    labels = result.boxes.cls.cpu().numpy().astype(int) + 1
                    scores = result.boxes.conf.cpu().numpy()
                    
                    # 确保mask尺寸匹配
                    resized_masks = []
                    for mask in masks_np:
                        resized_mask = resize_mask_to_target(mask, (args.mask_size, args.mask_size))
                        resized_masks.append(resized_mask)
                    
                    if len(resized_masks) > 0:
                        masks_final = np.stack(resized_masks, axis=0)
                        # MRCNN输出格式是 [N, 1, H, W]，添加channel维度
                        masks_final = masks_final[:, np.newaxis, :, :]
                    else:
                        masks_final = np.zeros((0, 1, args.mask_size, args.mask_size))
                else:
                    masks_final = np.zeros((0, 1, args.mask_size, args.mask_size))
                    labels = np.array([], dtype=int)
                    scores = np.array([])
                
                all_pred.append({
                    'label': labels,
                    'mask': masks_final,
                    'score': scores,
                })
        
        # 计算指标（与MRCNN使用相同的函数）
        precision, recall, precision_per_class, recall_per_class = utils.get_instance_segmentation_metrics(
            all_gt, all_pred
        )
        
        logger.info(f'=================== Split [{split}] Segmentation Result =====================')
        for i, name in enumerate(ALFRED_INTEREST_OBJECTS):
            class_idx = i + 1
            if class_idx in precision_per_class or class_idx in recall_per_class:
                p = precision_per_class.get(class_idx, 'nan')
                r = recall_per_class.get(class_idx, 'nan')
                logger.info(f'{split} {class_idx:03d}-{name:20s} Precision {p:20s} Recall {r:20s}')
        logger.info(f'{split} Precision {precision:.2f} Recall {recall:.2f}')
        
        results_all[split] = {
            'precision': precision,
            'recall': recall,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
        }
    
    return results_all


def main():
    args = arguments.parse_args()
    utils.set_random_seed(args.seed)
    
    args.name = 'evaluate_yolo'
    logging_dir = os.path.join('logs', args.name)
    os.makedirs(logging_dir, exist_ok=True)
    
    logger = utils.build_logger(filepath=os.path.join(logging_dir, 'log.txt'), verbose=True)
    
    if not YOLO_AVAILABLE:
        logger.error('ultralytics not installed. Please run: pip install ultralytics')
        return
    
    # 加载模型
    assert args.resume is not None, 'Please specify model path with --resume'
    logger.info(f'Loading model from: {args.resume}')
    
    model = YOLO(args.resume)
    
    # 设置数据加载参数
    args.num_workers = args.num_workers // 8 if args.num_workers > 8 else args.num_workers
    
    # 评估
    results = evaluate_yolo_on_alfred(
        model, args, logger, 
        splits=['valid_seen', 'valid_unseen']
    )
    
    # 汇总结果
    logger.info('\n=================== Summary ==================')
    for split, metrics in results.items():
        logger.info(f'{split}: Precision={metrics["precision"]:.2f}, Recall={metrics["recall"]:.2f}')


if __name__ == '__main__':
    main()
