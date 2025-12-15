"""
YOLO实例分割模型训练脚本
使用ultralytics YOLO库训练ALFRED数据集的实例分割模型
"""
import sys
import os
# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

import torch
import numpy as np
from utils import utils, arguments
from utils.alfred_interest_objects import ALFRED_INTEREST_OBJECTS
import logging
from tensorboardX import SummaryWriter
from datasets.yolo_segmentation_dataset import (
    export_yolo_dataset, create_yolo_yaml, AlfredYOLODataset
)

# 检查是否安装了ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Please run: pip install ultralytics")

torch.multiprocessing.set_sharing_strategy('file_system')


def prepare_yolo_dataset(args, data_root='./data'):
    """准备YOLO格式的数据集"""
    yolo_data_dir = os.path.join(data_root, 'yolo_format')
    
    # 检查是否已经导出
    yaml_path = os.path.join(yolo_data_dir, 'alfred_yolo.yaml')
    if os.path.exists(yaml_path):
        print(f'YOLO dataset already exists at {yolo_data_dir}')
        return yaml_path
    
    print('Preparing YOLO format dataset...')
    for split in ['train', 'valid_seen', 'valid_unseen']:
        print(f'  Exporting {split}...')
        export_yolo_dataset(args, split, yolo_data_dir, data_root)
    
    yaml_path = create_yolo_yaml(yolo_data_dir, data_root)
    return yaml_path


def train_yolo_native(args, yaml_path, logger):
    """使用ultralytics原生训练方式"""
    if not YOLO_AVAILABLE:
        raise ImportError("ultralytics not installed. Please run: pip install ultralytics")
    
    # 选择模型规模
    model_scale = getattr(args, 'model_scale', 'n')  # n, s, m, l, x
    
    # 加载预训练模型或自定义配置
    if args.model_config:
        # 使用自定义配置文件
        model = YOLO(args.model_config)
        logger.info(f'Using custom model config: {args.model_config}')
    elif args.resume:
        # 从检查点恢复
        model = YOLO(args.resume)
        logger.info(f'Resuming from: {args.resume}')
    else:
        # 使用预训练模型
        model_name = f'yolo11{model_scale}-seg.pt'  # YOLO11 segmentation
        model = YOLO(model_name)
        logger.info(f'Using pretrained model: {model_name}')
    
    # 训练参数
    train_args = {
        'data': yaml_path,
        'epochs': args.epoch,
        'imgsz': getattr(args, 'image_size', 640),
        'batch': args.bz,
        'device': args.gpu[0] if len(args.gpu) == 1 else args.gpu,
        'workers': args.num_workers,
        'project': 'logs',
        'name': args.name,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': args.optimizer,
        'lr0': args.base_lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_step / 1000 if args.warmup_step > 100 else 3,
        'cos_lr': args.cosine,
        'amp': args.amp,
        'seed': args.seed,
        'val': True,
        'save_period': args.save_freq,
        'plots': True,
        'verbose': True,
    }
    
    # 如果是恢复训练
    if args.resume:
        train_args['resume'] = True
    
    logger.info(f'Training arguments: {train_args}')
    
    # 开始训练
    results = model.train(**train_args)
    
    return model, results


def train_yolo_custom(args, logger):
    """
    使用自定义训练循环（与train_mrcnn.py类似的风格）
    这种方式允许更细粒度的控制，但需要手动实现训练逻辑
    """
    from datasets.yolo_segmentation_dataset import build_yolo_dataloader
    from datasets.segmentation_dataset import build_seg_dataloader
    
    device = torch.device(f'cuda:{args.gpu[0]}')
    
    # 构建数据加载器
    args.image_size = getattr(args, 'image_size', 640)
    train_dataloader = build_yolo_dataloader('train', args)
    vs_dataloader = build_yolo_dataloader('valid_seen', args)
    vu_dataloader = build_yolo_dataloader('valid_unseen', args)
    
    logger.info(f'Train: {len(train_dataloader.dataset)} images, {len(train_dataloader)} batches')
    logger.info(f'Valid Seen: {len(vs_dataloader.dataset)} images')
    logger.info(f'Valid Unseen: {len(vu_dataloader.dataset)} images')
    
    # 这里只是示例框架，实际的YOLO自定义训练需要更复杂的实现
    # 建议使用ultralytics原生训练方式
    logger.warning('Custom training loop is not fully implemented. '
                   'Please use --native flag for ultralytics native training.')
    

def evaluate_yolo_model(model, args, logger):
    """评估YOLO模型并计算与MRCNN相同的指标"""
    from datasets.segmentation_dataset import build_seg_dataloader
    
    device = torch.device(f'cuda:{args.gpu[0]}')
    
    # 使用原始数据集格式进行评估，以便与MRCNN结果对比
    args.image_size = 300
    args.mask_size = 300
    vs_dataloader = build_seg_dataloader('AlfredSegImageDataset', split='valid_seen', args=args)
    vu_dataloader = build_seg_dataloader('AlfredSegImageDataset', split='valid_unseen', args=args)
    
    for split, dataloader in [('valid_seen', vs_dataloader), ('valid_unseen', vu_dataloader)]:
        logger.info(f'=================== {split} Evaluation Start ==================')
        all_pred, all_gt = [], []
        
        for i, blobs in enumerate(dataloader):
            images, targets = blobs
            if len(images) == 0:
                continue
            
            # YOLO推理
            results = model.predict(
                [img.permute(1, 2, 0).numpy() * 255 for img in images],
                imgsz=args.image_size,
                conf=0.5,
                device=device,
                verbose=False,
            )
            
            # 收集GT
            for tgt in targets:
                all_gt.append({
                    'label': tgt['labels'].numpy(),
                    'mask': tgt['masks'].numpy()
                })
            
            # 收集预测结果
            for result in results:
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    labels = result.boxes.cls.cpu().numpy().astype(int) + 1  # YOLO从0开始，MRCNN从1开始
                    scores = result.boxes.conf.cpu().numpy()
                else:
                    masks = np.zeros((0, args.mask_size, args.mask_size))
                    labels = np.array([])
                    scores = np.array([])
                
                all_pred.append({
                    'label': labels,
                    'mask': masks[:, np.newaxis, :, :] if len(masks) > 0 else masks,  # [N, 1, H, W]
                    'score': scores,
                })
        
        # 计算指标
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


def main():
    args = arguments.parse_args()
    utils.set_random_seed(args.seed)
    
    if args.name is None:
        args.name = 'train_yolo_seg'
    
    logging_dir = os.path.join('logs', args.name)
    os.makedirs(logging_dir, exist_ok=True)
    
    logger = utils.build_logger(filepath=os.path.join(logging_dir, 'log.txt'), verbose=True)
    
    # 设置默认参数
    if not hasattr(args, 'image_size') or args.image_size == 300:
        args.image_size = 640  # YOLO默认输入尺寸
    
    logger.info(f'Training YOLO Segmentation Model')
    logger.info(f'Args: {args}')
    
    # 准备数据集
    yaml_path = prepare_yolo_dataset(args)
    logger.info(f'Dataset config: {yaml_path}')
    
    # 使用ultralytics原生训练
    if YOLO_AVAILABLE:
        model, results = train_yolo_native(args, yaml_path, logger)
        
        # 保存最终模型
        final_model_path = os.path.join(logging_dir, 'best.pt')
        logger.info(f'Best model saved to: {final_model_path}')
        
        # 评估模型（使用与MRCNN相同的指标）
        logger.info('Evaluating with ALFRED metrics...')
        evaluate_yolo_model(model, args, logger)
    else:
        logger.error('ultralytics not installed. Please run: pip install ultralytics')
        return


if __name__ == '__main__':
    main()
