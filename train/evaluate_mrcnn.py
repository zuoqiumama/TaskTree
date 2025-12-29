import cv2
import numpy as np
import torch,torchvision
import sys
import os
# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (假设 train/scene_gen/ 是两层深度，根据实际情况调整 '../..')
project_root = os.path.abspath(os.path.join(current_dir, "../"))
# 将根目录加入系统路径
sys.path.append(project_root)
from models.segmentation.sam.sam_wrapper import SamSegTrainWrapper
from models.segmentation.sam.sam_dp_wrapper import SamDataParallelWrapper
from models.segmentation.maskrcnn.custom_maskrcnn import get_custom_maskrcnn
from datasets.segmentation_dataset import build_seg_dataloader
from utils import utils, arguments
import tqdm
import torch.nn.functional as F
import cv2,time
from utils.optimizer import build_lr_scheduler,build_optimizer
from itertools import chain
from collections import OrderedDict
from utils.alfred_interest_objects import ALFRED_INTEREST_OBJECTS
import os
import logging
from torch.nn.parallel._functions import Scatter
from tensorboardX import SummaryWriter
from train_mrcnn import MRCNNDataParallelWrapper
torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    args = arguments.parse_args()
    utils.set_random_seed(args.seed)

    args.name = __file__.split('/')[-1].split('.')[0]
    logging_dir = os.path.join('logs', args.name)
    os.makedirs(logging_dir,exist_ok=True)

    logger = utils.build_logger(filepath=os.path.join(logging_dir,'log.txt'),verbose=True)
    device = torch.device('cuda:%d' % args.gpu[0])
    
    use_dp = len(args.gpu) > 1
    if use_dp:
        args.bz = args.bz * len(args.gpu)
    args.image_size  = 300
    args.mask_size   = 300
    args.num_workers = args.num_workers // 8
    vs_dataloader = build_seg_dataloader('AlfredSegImageDataset', split='valid_seen', args=args)
    vu_dataloader = build_seg_dataloader('AlfredSegImageDataset', split='valid_unseen', args=args)
    print('>>> ValidSeen #Dataset %d #Dataloader %d' % (len(vs_dataloader.dataset), len(vs_dataloader)  ))
    print('>>> ValidUnseen #Dataset %d #Dataloader %d' % (len(vu_dataloader.dataset), len(vu_dataloader)  ))
    
    num_classes = vs_dataloader.dataset.n_obj + 1
    model = get_custom_maskrcnn(num_classes=num_classes, 
                                use_cbam=args.use_cbam, 
                                use_scconv=args.use_scconv, 
                                use_proto=args.use_proto, 
                                use_csa=args.use_csa,
                                box_score_thresh=0.5,
                                pretrained_backbone=False)
    
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(
    #     weights = None, weights_backbone = None, num_classes=vs_dataloader.dataset.n_obj + 1, box_score_thresh = 0.5)
    

    model = model.to(device)
    if use_dp:
        model = MRCNNDataParallelWrapper(model,device_ids=args.gpu)

    
    
    assert args.resume is not None
    logger.info('Load Resume from <===== %s' % args.resume)
    resume = torch.load(args.resume,map_location='cpu')
    if use_dp:
        model.module.load_state_dict(resume['model'])
    else:
        model.load_state_dict(resume['model'])
        
    model.eval()
    for split, dataloader in [('valid_seen', vs_dataloader), ('valid_unseen', vu_dataloader)]:
        
        logger.info(f'=================== {split} Test Start ==================')
        all_pred_match, all_gt_match = [], []
        all_pred_class, all_gt_class = [], []
        
        pbar = tqdm.tqdm(dataloader, desc=f'{split} Eval')
        for i, blobs in enumerate(pbar):
            images, targets = blobs
            if len(images) == 0:
                continue
            images  = [img.to(device) for img in images]
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.amp):
                    output = model(images)
            
            for j in range(len(targets)):
                tgt = targets[j]
                pred = output[j]
                
                gts = {
                    'label' : tgt['labels'].numpy(),
                    'mask'  : tgt['masks'].numpy()
                }
                preds = {
                    'label' : pred['labels'].data.cpu().numpy(),
                    'mask'  : pred['masks'].data.cpu().numpy(),
                    'score' : pred['scores'].data.cpu().numpy(),
                }
                
                pred_match, gt_match = utils.instance_match((gts, preds, 0.5))
                
                all_pred_match.append(pred_match)
                all_gt_match.append(gt_match)
                all_pred_class.append(preds['label'])
                all_gt_class.append(gts['label'])

        logger.info(f'=================== {split} Inference Done ==================')
        
        pred_match = np.concatenate(all_pred_match, 0) if all_pred_match else np.array([])
        gt_match   = np.concatenate(all_gt_match, 0) if all_gt_match else np.array([])
        pred_class = np.concatenate(all_pred_class, 0) if all_pred_class else np.array([])
        gt_class   = np.concatenate(all_gt_class, 0) if all_gt_class else np.array([])
        
        recall    = gt_match.mean() * 100 if len(gt_match) > 0 else 0
        precision = pred_match.mean() * 100 if len(pred_match) > 0 else 0
        
        precision_per_class = {}
        recall_per_class = {}
        
        for i in np.unique(gt_class):
            instances = gt_match[gt_class == i]
            recall_per_class[i] = '%d / %d = %.2f' % (instances.sum(),instances.shape[0],instances.mean()*100) + '%'
        for i in np.unique(pred_class):
            if i < 0:
                continue
            instances = pred_match[pred_class == i]
            precision_per_class[i] = '%d / %d = %.2f' % (instances.sum(),instances.shape[0],instances.mean()*100) + '%'
        logger.info(f'=================== Split [{split}] Segmentation Result =====================')
        for i, name in enumerate(ALFRED_INTEREST_OBJECTS):
            class_idx = i + 1
            if class_idx in precision_per_class or class_idx in recall_per_class:
                p = precision_per_class.get(class_idx,'nan')
                r = recall_per_class.get(class_idx,'nan')
                logger.info(f'{split} {class_idx:03d}-{name:20s} Precision {p:20s} Recall {r:20s}')
        logger.info(f'{split} Precision {precision:.2f} Recall {recall:.2f}')

if __name__ == '__main__':
    main()
