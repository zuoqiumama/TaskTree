"""
YOLO格式的分割数据集
将ALFRED数据集转换为YOLO实例分割格式用于训练
"""
import torch
import json
import os
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
import sys
from utils.utils import remove_alfred_repeat_tasks
import cv2
from utils.alfred_interest_objects import ALFRED_INTEREST_OBJECTS
from torchvision.transforms.functional import resize, to_pil_image
import shutil


class AlfredYOLODataset(Dataset):
    """
    将ALFRED数据集转换为YOLO训练格式的数据集
    该数据集可以直接用于ultralytics YOLO训练
    """
    def __init__(self, args, split, data_root='./data') -> None:
        super().__init__()
        self.args = args
        self.split = split
        self.data_root = data_root
        self.tasks = json.load(open(os.path.join(data_root, 'splits/oct21.json')))[split]
        self.tasks = remove_alfred_repeat_tasks(self.tasks)
        self.images = []
        for i, t in enumerate(self.tasks):
            rgb_dir = os.path.join(data_root, 'replay', split, t['task'], 'rgb')
            if os.path.exists(rgb_dir):
                n_frame = len(os.listdir(rgb_dir))
                for j in range(n_frame):
                    self.images.append([t['task'], j])

        self.image_size = getattr(args, 'image_size', 640)
        
        # class = 0,1,2, .... object (YOLO从0开始)
        self.object2class = {
            obj: i for i, obj in enumerate(ALFRED_INTEREST_OBJECTS)
        }
        self.n_obj = len(self.object2class)
        
        print(f'[YOLODataset] Split = {split}. Tasks = {len(self.tasks)}. Images = {len(self.images)}. Objects = {self.n_obj}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        task, step = self.images[index]
        rgb_path = os.path.join(self.data_root, 'replay', self.split, task, 'rgb', 'step_%05d.png' % step)
        seg_path = os.path.join(self.data_root, 'replay', self.split, task, 'seg', 'step_%05d.pkl' % step)
        
        # 读取图像
        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        image = np.array(resize(to_pil_image(image), (self.image_size, self.image_size)))
        
        # 读取分割信息
        seg_info = pkl.load(open(seg_path, 'rb'))
        
        bboxes = []  # [class_id, x_center, y_center, width, height] 归一化
        segments = []  # polygon points 归一化
        masks = []  # binary masks
        
        for seg in seg_info['object']:
            objectType = seg['objectType']
            if objectType not in self.object2class:
                continue
            objectClass = self.object2class[objectType]
            mask = (seg_info['mask'] == seg['label']).astype(np.uint8)
            
            if mask.sum() < 10:  # 忽略太小的物体
                continue
            
            # 计算边界框 (归一化)
            x1, y1, x2, y2 = self.mask_to_box(mask)
            x_center = (x1 + x2) / 2.0 / orig_w
            y_center = (y1 + y2) / 2.0 / orig_h
            width = (x2 - x1) / orig_w
            height = (y2 - y1) / orig_h
            
            # 提取轮廓点用于分割
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            # 取最大轮廓
            contour = max(contours, key=cv2.contourArea)
            if len(contour) < 3:
                continue
            
            # 简化轮廓点并归一化
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 3:
                approx = contour
            
            polygon = []
            for point in approx.reshape(-1, 2):
                polygon.extend([point[0] / orig_w, point[1] / orig_h])
            
            bboxes.append([objectClass, x_center, y_center, width, height])
            segments.append(polygon)
            
            # resize mask
            mask_resized = np.array(resize(to_pil_image(mask * 255), (self.image_size, self.image_size)))
            masks.append(mask_resized > 0)
        
        # 转换为tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if len(masks) > 0:
            masks = torch.from_numpy(np.stack(masks, 0)).float()
            bboxes = torch.tensor(bboxes, dtype=torch.float32)
        else:
            masks = torch.zeros((0, self.image_size, self.image_size), dtype=torch.float32)
            bboxes = torch.zeros((0, 5), dtype=torch.float32)
        
        return {
            'image': image,
            'bboxes': bboxes,  # [N, 5]: class_id, x_center, y_center, w, h (normalized)
            'masks': masks,    # [N, H, W]
            'segments': segments,  # list of polygon points
            'task': task,
            'step': step,
            'image_path': rgb_path,
        }

    def mask_to_box(self, mask):
        h, w = mask.shape
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        xs = xs[mask > 0]
        ys = ys[mask > 0]
        if len(xs) == 0:
            return [0, 0, 1, 1]
        return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]

    @staticmethod
    def collate_fn(batch):
        """自定义collate函数用于YOLO训练"""
        images = []
        targets = []
        
        for i, item in enumerate(batch):
            images.append(item['image'])
            
            # 为每个目标添加batch index
            bboxes = item['bboxes']
            if len(bboxes) > 0:
                batch_idx = torch.full((len(bboxes), 1), i, dtype=torch.float32)
                target = torch.cat([batch_idx, bboxes], dim=1)
                targets.append(target)
        
        images = torch.stack(images, 0)
        if len(targets) > 0:
            targets = torch.cat(targets, 0)
        else:
            targets = torch.zeros((0, 6), dtype=torch.float32)
        
        return {
            'images': images,
            'targets': targets,
            'batch': batch,
        }


def export_yolo_dataset(args, split, output_dir, data_root='./data'):
    """
    导出YOLO格式的数据集到磁盘
    生成YOLO所需的images和labels目录结构
    """
    dataset = AlfredYOLODataset(args, split, data_root)
    
    images_dir = os.path.join(output_dir, 'images', split)
    labels_dir = os.path.join(output_dir, 'labels', split)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f'Exporting {split} dataset to {output_dir}...')
    
    for idx in range(len(dataset)):
        item = dataset[idx]
        task = item['task'].replace('/', '_')
        step = item['step']
        
        # 保存图像
        img_name = f'{task}_step_{step:05d}.jpg'
        img_path = os.path.join(images_dir, img_name)
        
        # 复制原图像（不缩放，让YOLO自己处理）
        src_img_path = item['image_path']
        img = cv2.imread(src_img_path)
        cv2.imwrite(img_path, img)
        
        # 保存标签 (YOLO分割格式)
        label_name = f'{task}_step_{step:05d}.txt'
        label_path = os.path.join(labels_dir, label_name)
        
        with open(label_path, 'w') as f:
            for bbox, segment in zip(item['bboxes'].numpy(), item['segments']):
                class_id = int(bbox[0])
                # YOLO分割格式: class_id x1 y1 x2 y2 ... (polygon points)
                line = f"{class_id}"
                for coord in segment:
                    line += f" {coord:.6f}"
                f.write(line + '\n')
        
        if idx % 1000 == 0:
            print(f'  Processed {idx}/{len(dataset)} images')
    
    print(f'  Done. Total {len(dataset)} images exported.')


def create_yolo_yaml(output_dir, data_root='./data'):
    """创建YOLO数据集配置文件"""
    yaml_content = f"""# ALFRED Instance Segmentation Dataset for YOLO
# Auto-generated for DISCO project

path: {os.path.abspath(output_dir)}
train: images/train
val: images/valid_seen
test: images/valid_unseen

# Number of classes
nc: {len(ALFRED_INTEREST_OBJECTS)}

# Class names
names:
"""
    for i, obj in enumerate(ALFRED_INTEREST_OBJECTS):
        yaml_content += f"  {i}: {obj}\n"
    
    yaml_path = os.path.join(output_dir, 'alfred_yolo.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f'Created YOLO config: {yaml_path}')
    return yaml_path


def build_yolo_dataloader(split, args, data_root='./data'):
    """构建YOLO格式的DataLoader"""
    dataset = AlfredYOLODataset(args, split, data_root)
    dataloader = DataLoader(
        dataset,
        batch_size=args.bz,
        shuffle=(split == 'train'),
        num_workers=args.num_workers,
        collate_fn=AlfredYOLODataset.collate_fn,
        pin_memory=True,
    )
    return dataloader


if __name__ == '__main__':
    """测试数据集导出"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./data/yolo_format')
    parser.add_argument('--image_size', type=int, default=640)
    parser.add_argument('--splits', type=str, nargs='+', 
                        default=['train', 'valid_seen', 'valid_unseen'])
    args = parser.parse_args()
    
    for split in args.splits:
        export_yolo_dataset(args, split, args.output_dir)
    
    create_yolo_yaml(args.output_dir)
