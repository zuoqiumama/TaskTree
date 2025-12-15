# YOLO实例分割模型

本目录包含用于ALFRED数据集的YOLO实例分割模型相关文件。

## 文件说明

- `yolo11-seg.yaml` - YOLO11分割模型配置文件
- `yolov8-seg.yaml` - YOLOv8分割模型配置文件
- `custom_modules.py` - 自定义模块（CBAM、SCConv等注意力机制）
- `yolo_wrapper.py` - YOLO模型封装器，提供与Mask R-CNN兼容的接口

## 安装依赖

```bash
pip install ultralytics
```

## 训练模型

### 1. 准备数据集

首次运行会自动将ALFRED数据集转换为YOLO格式：

```bash
# 直接运行训练脚本会自动准备数据
python train/train_yolo.py --name yolo_seg_exp --epoch 100 --bz 16 --gpu 0
```

或者手动导出数据集：

```bash
python datasets/yolo_segmentation_dataset.py --output_dir ./data/yolo_format
```

### 2. 开始训练

```bash
# 使用YOLO11-n (最小模型)
python train/train_yolo.py \
    --name yolo11n_seg \
    --epoch 100 \
    --bz 16 \
    --gpu 0 \
    --base_lr 0.01

# 使用YOLO11-s (小模型)
python train/train_yolo.py \
    --name yolo11s_seg \
    --epoch 100 \
    --bz 8 \
    --gpu 0 \
    --model_scale s

# 从检查点恢复训练
python train/train_yolo.py \
    --name yolo11n_seg \
    --resume logs/yolo11n_seg/weights/last.pt
```

### 3. 使用自定义配置

```bash
# 使用自定义YOLO配置文件
python train/train_yolo.py \
    --name yolo_custom \
    --model_config models/segmentation/yolo/yolo11-seg.yaml \
    --epoch 100 \
    --bz 16
```

## 评估模型

使用与Mask R-CNN相同的评估指标：

```bash
python train/evaluate_yolo.py \
    --resume logs/yolo11n_seg/weights/best.pt \
    --gpu 0 \
    --bz 16
```

## 在智能体中使用

```python
from models.segmentation.yolo.yolo_wrapper import load_yolo_seg_model

# 加载模型
model = load_yolo_seg_model('weights/yolo_seg.pt', device='cuda:0')
model.eval()

# 预测
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
output = model.predict_single(image)

# output包含:
# - boxes: [N, 4] 边界框
# - labels: [N] 类别ID (从1开始)
# - scores: [N] 置信度
# - masks: [N, 1, H, W] 分割mask
```

## 添加自定义模块

可以在`custom_modules.py`中添加自定义注意力模块，然后在YOLO配置文件中使用：

```yaml
# 在yolo11-seg.yaml中使用CBAM模块
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, CBAM, [64]]  # 添加CBAM注意力
  ...
```

## 模型对比

| 模型 | Precision (VS) | Recall (VS) | Precision (VU) | Recall (VU) |
|------|----------------|-------------|----------------|-------------|
| Mask R-CNN | - | - | - | - |
| YOLO11-n | - | - | - | - |
| YOLO11-s | - | - | - | - |

（数值待训练后填写）
