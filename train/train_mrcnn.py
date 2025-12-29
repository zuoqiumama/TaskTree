import sys
import os
# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(project_root)

import cv2
import numpy as np
import torch,torchvision
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
torch.multiprocessing.set_sharing_strategy('file_system')

class MRCNNDataParallelWrapper(torch.nn.DataParallel):
    def scatter(
        self,
        inputs,
        kwargs,
        device_ids,
    ):
        n_dev = len(device_ids)
        n_bz = len(inputs[0])
        size = [ (i + 1) * n_bz // n_dev  - i * n_bz // n_dev  for i in range(n_dev) ]
        size = [ s for s in size if s > 0]

        cur = 0
        new_inputs = []
        for i,s in enumerate(size):
            device = torch.device('cuda:%d' % device_ids[i])
            if len(inputs) == 1:
                images = inputs[0]
                for i in range(cur,cur+s):
                    images[i] = images[i].to(device)
                new_inputs.append((images[cur:cur+s],))
            else:
                images, targets = inputs
                for i in range(cur,cur+s):
                    images[i] = images[i].to(device)
                    for k in targets[i]:
                        if isinstance(targets[i][k],torch.Tensor):
                            targets[i][k] = targets[i][k].to(device)
                new_inputs.append((images[cur:cur+s], targets[cur:cur+s]))
            cur += s
        kwargs = [kwargs for _ in range(len(new_inputs))]
        return new_inputs,kwargs

    def gather(self, outputs, output_device):
        device = torch.device('cuda:%d' % output_device)
        if isinstance(outputs[0],list):
            final_out = []
            for output in outputs:
                for img_result in output:
                    for k in img_result:
                        img_result[k] = img_result[k].to(device)
                    final_out.append(img_result)
            return final_out
        elif isinstance(outputs[0],dict):
            final_out = {}
            keys = list(outputs[0].keys())
            for k in keys:
                assert 'loss' in k
                final_out[k] = [output[k].to(device) for output in outputs]
                final_out[k] = sum(final_out[k]) / len(final_out[k])
            return final_out
        else:
            raise NotImplementedError    
        



def main():
    args = arguments.parse_args()
    utils.set_random_seed(args.seed)

    logging_dir = os.path.join('logs', args.name)
    os.makedirs(logging_dir,exist_ok=True)

    logger = utils.build_logger(filepath=os.path.join(logging_dir,'log.txt'),verbose=True)
    writer = SummaryWriter(os.path.join(logging_dir,'tb'))
    device = torch.device('cuda:%d' % args.gpu[0])
    
    use_dp = len(args.gpu) > 1
    if use_dp:
        args.bz = args.bz * len(args.gpu)
    args.image_size  = 300
    args.mask_size   = 300
    train_dataloader = build_seg_dataloader('AlfredSegImageDataset', split='train', args=args)
    args.num_workers = args.num_workers // 8
    vs_dataloader = build_seg_dataloader('AlfredSegImageDataset', split='valid_seen', args=args)
    vu_dataloader = build_seg_dataloader('AlfredSegImageDataset', split='valid_unseen', args=args)
    print('>>> Train #Dataset %d #Dataloader %d' % (len(train_dataloader.dataset), len(train_dataloader) ))
    print('>>> ValidSeen #Dataset %d #Dataloader %d' % (len(vs_dataloader.dataset), len(vs_dataloader)  ))
    print('>>> ValidUnseen #Dataset %d #Dataloader %d' % (len(vu_dataloader.dataset), len(vu_dataloader)  ))
    
    num_classes = train_dataloader.dataset.n_obj + 1
    model = get_custom_maskrcnn(num_classes=num_classes, 
                                use_cbam=args.use_cbam, 
                                use_scconv=args.use_scconv, 
                                use_proto=args.use_proto, 
                                use_csa=args.use_csa,
                                box_score_thresh=0.5)
    
    pretrained_keys = set()
    if args.pretrained_path:
        print(f'Loading pretrained weights from {args.pretrained_path}')
        ckpt = torch.load(args.pretrained_path, map_location='cpu')
        if 'model' in ckpt:
            state_dict = ckpt['model']
        else:
            state_dict = ckpt
            
        model_dict = model.state_dict()
        # Filter out unnecessary keys and keys with shape mismatch
        pretrained_dict = {}
        for k, v in state_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    pretrained_dict[k] = v
                    pretrained_keys.add(k)
                else:
                    print(f'Skipping {k} due to shape mismatch: {v.shape} vs {model_dict[k].shape}')
            else:
                # print(f'Skipping {k} as it is not in the model')
                pass
                
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f'Loaded {len(pretrained_dict)}/{len(model_dict)} keys from pretrained weights')
        
        if args.freeze_epochs > 0:
            print(f'Freezing {len(pretrained_keys)} pretrained parameters for {args.freeze_epochs} epochs')
            for name, param in model.named_parameters():
                if name in pretrained_keys:
                    param.requires_grad = False

    model = model.to(device)
    if use_dp:
        model = MRCNNDataParallelWrapper(model,device_ids=args.gpu)

    # Separate parameters into pretrained (low LR) and new (base LR) groups
    pretrained_params = []
    new_params = []
    
    for name, param in model.named_parameters():
        # Handle DataParallel prefix
        key_name = name
        if use_dp and name.startswith('module.'):
            key_name = name[7:]
            
        if key_name in pretrained_keys:
            pretrained_params.append(param)
        else:
            new_params.append(param)
            
    print(f"Optimizer groups: {len(pretrained_params)} pretrained params (lr={args.base_lr*0.1:.2e}), {len(new_params)} new params (lr={args.base_lr:.2e})")

    param_groups = [
        {'params': pretrained_params, 'lr': args.base_lr * 0.1},
        {'params': new_params, 'lr': args.base_lr}
    ]

    optimizer = build_optimizer(args, param_groups)
    lr_scheduler = build_lr_scheduler(args,optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    if args.resume is not None:
        print('Load Resume from <===== %s' % args.resume)
        resume = torch.load(args.resume,map_location='cpu')
        # for pg in resume['optimizer']['param_groups']:
        #     pg['lr'] = args.base_lr
        optimizer.load_state_dict(resume['optimizer'])
        # lr_scheduler.load_state_dict(resume['lr_scheduler'])
        if use_dp:
            model.module.load_state_dict(resume['model'])
        else:
            model.load_state_dict(resume['model'])
        
    
    for ep in range(1, args.epoch + 1):
        if args.freeze_epochs > 0 and ep == args.freeze_epochs + 1:
            print(f'Unfreezing pretrained parameters at epoch {ep}')
            for name, param in model.named_parameters():
                if name in pretrained_keys:
                    param.requires_grad = True
                    
        logger.info(f'=================== Epoch {ep} Train Start ==================')
        model.train()
        pbar = tqdm.tqdm(train_dataloader, desc=f'Epoch {ep}')
        for i, blobs in enumerate(pbar):
            step = 1 + i + (ep - 1) * len(train_dataloader)
            images, targets = blobs
            images  = [img.to(device) for img in images]
            targets = [{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in t.items()} for t in targets]
            # print(targets[0]['boxes'])
            with torch.cuda.amp.autocast(enabled=args.amp):
                output = model(images, targets)  # Returns losses and detections
            loss_weights = {
                'loss_classifier' : 1,
                'loss_box_reg' : 1,
                'loss_mask' : 1,
                'loss_objectness' : 5,
                'loss_rpn_box_reg' : 1,
                'loss_proto': 0.001,
            }

            output = {k:loss_weights.get(k, 1.0) * v for k,v in output.items()}
            loss = sum(output.values())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            for key,val in output.items():
                writer.add_scalar(key, val, step)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f"{optimizer.param_groups[0]['lr']:.2e}"})
            
            if step % args.report_freq == 0 or i == len(train_dataloader) - 1:
                report = 'Epoch %.4f Step %d. LR %.2e  loss %.4f.' % (
                    step/len(train_dataloader), step , optimizer.param_groups[0]['lr'], loss,
                )
                logger.info(report)
        logger.info(f'=================== Epoch {ep} Train Done ==================\n')

        if ep % args.test_freq == 0:
            logger.info(f'=================== Epoch {ep} Test Start ==================')
            model.eval()
            # for split, dataloader in [('valid_seen', vs_dataloader), ('valid_unseen', vu_dataloader)]:
            for split, dataloader in [('valid_unseen', vu_dataloader)]:            
                all_pred_match, all_gt_match = [], []
                all_pred_class, all_gt_class = [], []
                
                for i, blobs in enumerate(dataloader):
                    images, targets = blobs
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

            save_path = os.path.join(logging_dir, 'epoch_%d.pth' % ep)
            logger.info('Save Resume to ====> %s' % save_path)
            torch.save({
                'model' : model.state_dict() if not use_dp else model.module.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'lr_scheduler' : lr_scheduler.state_dict(),
                }, save_path)


if __name__ == '__main__':
    main()
