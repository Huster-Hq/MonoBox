import torch
import cv2
import numpy as np


def dice_coefficient(x, target, weight=None):
    # print(x.shape,target.shape)
    if weight is None:
        eps = 1e-5
        n_inst = x.size(0)
        x = x.reshape(n_inst, -1)
        target = target.reshape(n_inst, -1)
        intersection = (x * target).sum(dim=1)
        union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
        loss = 1. - (2 * intersection / union)
    else:
        x = x*weight
        target = target*weight
        eps = 1e-5
        n_inst = x.size(0)
        x = x.reshape(n_inst, -1)
        target = target.reshape(n_inst, -1)
        intersection = (x * target).sum(dim=1)
        union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
        loss = 1. - (2 * intersection / union)
    return loss


def structure_loss(pred, mask):  #pred:tensor[[[[]]]]   (b,c,h,w)   [16,1,256,256]
    maskx = mask
    maskx[maskx>=0.5] = 1.
    maskx[maskx<0.5] = 0
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter + 1)
    return wiou.mean()


# ---- Monotonicity Constraint (MC) ----
def MC_loss(ypred, box_mask, unconf=0.2):
    pred = ypred
    label = box_mask
    b,_,img_h,img_w = box_mask.shape
    box_mask_np = box_mask.clone().cpu().numpy()

    projx = label.max(dim=2, keepdim=True)[0]
    projy = label.max(dim=3, keepdim=True)[0]
    
    pre_projx = pred.max(dim=2, keepdim=True)[0]  #[20,1,1,256]
    pre_projy = pred.max(dim=3, keepdim=True)[0]  #[20,1,256,1]

    for index in range(b):
        contours, _ = cv2.findContours(box_mask_np[index,:,:,:].squeeze(0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            xin_left = x
            xin_right = x+w-1
            yin_up = y
            yin_bottom = y+h-1
            confusion_w = min(int(w*unconf), x)
            confusion_h = min(int(h*unconf), y)
            if x+w+confusion_w > img_w:
                confusion_w = img_w-x-w
            if y+h+confusion_h > img_h:
                confusion_h = img_h-y-h

            pre_projx_ori = pre_projx.clone()
            pre_projy_ori = pre_projy.clone()

            if confusion_w>0:
                pre_projx[index,:,:,xin_left-confusion_w+1:xin_left+confusion_w] = 1 - torch.relu(pre_projx_ori[index,:,:,xin_left-confusion_w:xin_left+confusion_w-1] - pre_projx_ori[index,:,:,xin_left-confusion_w+1:xin_left+confusion_w] - 0.05)
                pre_projx[index,:,:,xin_right-confusion_w+1:xin_right+confusion_w] = 1 - torch.relu(pre_projx_ori[index,:,:,xin_right-confusion_w+2:xin_right+confusion_w+1] - pre_projx_ori[index,:,:,xin_right-confusion_w+1:xin_right+confusion_w] - 0.05)
                # projx[index,:,xin_left-confusion_w:xin_left] = 1
                # projx[index,:,:,xin_right+1:xin_right+confusion_w+1] = 1
                projx[index,:,:,xin_left-confusion_w+1:xin_left+confusion_w] = 1
                projx[index,:,:,xin_right-confusion_w+1:xin_right+confusion_w] = 1
            if confusion_h>0:
                pre_projy[index,:,yin_up-confusion_h+1:yin_up+confusion_h,:] = 1 - torch.relu(pre_projy_ori[index,:,yin_up-confusion_h:yin_up+confusion_h-1,:] - pre_projy_ori[index,:,yin_up-confusion_h+1:yin_up+confusion_h,:] - 0.05)
                pre_projy[index,:,yin_bottom-confusion_h+1:yin_bottom+confusion_h,:] = 1 - torch.relu(pre_projy_ori[index,:,yin_bottom-confusion_h+2:yin_bottom+confusion_h+1,:] - pre_projy_ori[index,:,yin_bottom-confusion_h+1:yin_bottom+confusion_h,:] - 0.05)
                # projy[index,:,yin_up-confusion_h:yin_up,:] = 1
                # projy[index,:,yin_bottom+1:yin_bottom+confusion_h+1,:] = 1
                projy[index,:,yin_up-confusion_h+1:yin_up+confusion_h,:] = 1
                projy[index,:,yin_bottom-confusion_h+1:yin_bottom+confusion_h,:] = 1

    label_back_proj = torch.matmul(projy, projx)
    pre_back_proj = torch.matmul(pre_projy, pre_projx)
    final_result = torch.where(label_back_proj!=label, pred, pre_back_proj)

    dice = dice_coefficient(final_result, label)
    return (dice).mean()