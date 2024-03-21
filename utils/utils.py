import torch
import cv2

import numpy as np
from thop import profile
from thop import clever_format


def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay
        lr = param_group['lr']
    return lr


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.
    """
    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious

def mask2boxes(pred, name=None):
    pred = pred.data.cpu().numpy().squeeze() #[B,H,W]
    boxes = []
    H,W = pred.shape[1:]
    for i in range(pred.shape[0]):
        mask = pred[i]
        mask[mask<0.5] = 0
        mask[mask>0.5] = 1
        batch_boxes = []
        contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        if len(contours) != 0:
            for id, contour in enumerate(contours):
                x,y,w,h = cv2.boundingRect(contour)
                box = [x, y, x+w-1, y+h-1]
                batch_boxes.append(box)
        boxes.append(batch_boxes)
    return boxes   #[B,N,4]


def update_box(box_mask, mask_T, iou_thres=0.5):
    label = box_mask
    pseudo_label = mask_T
    correct_box_mask = torch.zeros_like(box_mask)
    batch,_,img_h,img_w = box_mask.shape
    box_mask2box = mask2boxes(label)  # mask -> box (xmin,ymin,xmax,ymax)
    pseudo_label2box = mask2boxes(pseudo_label)  # mask -> box (xmin,ymin,xmax,ymax)
    for index in range(batch):
        pseudo_boxes = torch.tensor(pseudo_label2box[index], device=mask_T.device)
        label_boxes = torch.tensor(box_mask2box[index], device=mask_T.device)
        if pseudo_boxes.numel() != 0:
            if label_boxes.numel() == 0:
                continue
            iou = bbox_overlaps(pseudo_boxes, label_boxes) #box_mask2box:[M,4], pseudo_boxes:[N,4] iou: [M,N]
            iou_gtbox, idx = iou.max(dim=1)  # iou_gtbox:
            idx_match = []
            correct_label_boxes = []
            for indexx in range(len(iou_gtbox)):
                if iou_gtbox[indexx] > iou_thres:
                    idx_match.append(idx[indexx])
                    # label_box_id = idx[indexx]
                    correct_label_boxes.append(pseudo_boxes[indexx])
            for i in range(len(label_boxes)):
                if i not in idx_match:
                    correct_label_boxes.append(label_boxes[i])
            for box in correct_label_boxes:
                [xmin, ymin, xmax, ymax] = [element for element in box]
                correct_box_mask[index,:,ymin:ymax+1,xmin:xmax+1] = 1
        else:
            correct_box_mask[index,:,:,:] = box_mask[index,:,:,:]
    return correct_box_mask


def merge_box(box_mask, mask_T, iou_thres=0.5):
    label = box_mask
    pseudo_label = mask_T
    correct_box_mask = torch.zeros_like(box_mask)
    batch,_,img_h,img_w = box_mask.shape
    box_mask2box = mask2boxes(label)  # mask -> box (xmin,ymin,xmax,ymax)
    pseudo_label2box = mask2boxes(pseudo_label)  # mask -> box (xmin,ymin,xmax,ymax)
    for index in range(batch):
        pseudo_boxes = torch.tensor(pseudo_label2box[index], device=mask_T.device)
        label_boxes = torch.tensor(box_mask2box[index], device=mask_T.device)
        if pseudo_boxes.numel() != 0:
            if label_boxes.numel() == 0:
                continue
            iou = bbox_overlaps(pseudo_boxes, label_boxes)  #box_mask2box:[M,4], pseudo_boxes:[N,4] iou: [M,N]
            iou_gtbox, idx = iou.max(dim=1)  #iou_gtbox:
            idx_match = []
            correct_label_boxes = []
            for indexx in range(len(iou_gtbox)):
                if iou_gtbox[indexx] > iou_thres:
                    idx_match.append(idx[indexx])
                    label_box_id = idx[indexx]
                    correct_label_boxes.append((pseudo_boxes[indexx]+label_boxes[label_box_id])//2)
            for i in range(len(label_boxes)):
                if i not in idx_match:
                    correct_label_boxes.append(label_boxes[i])
            for box in correct_label_boxes:
                [xmin, ymin, xmax, ymax] = [element for element in box]
                correct_box_mask[index,:,ymin:ymax+1,xmin:xmax+1] = 1
        else:
            correct_box_mask[index,:,:,:] = box_mask[index,:,:,:]
    return correct_box_mask
