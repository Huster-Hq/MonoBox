import numpy as np
import cv2
import os


def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    gt = gt > 128
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt

def IOU_dice(pred, gt, thresholds):
    threshold_Dice = np.zeros(len(thresholds))
    threshold_IoU = np.zeros(len(thresholds))
    for k, threshold in enumerate(thresholds):
        if threshold > 1:
            threshold = 1
        Label3 = np.zeros_like(gt)
        Label3[pred >= threshold] = 1
        NumRec = np.sum(Label3 == 1)
        NumNoRec = np.sum(Label3 == 0)
        LabelAnd = (Label3 == 1) & (gt == 1)
        NumAnd = np.sum(LabelAnd == 1)
        num_obj = np.sum(gt)
        num_pred = np.sum(Label3)
        FN = num_obj - NumAnd
        FP = NumRec - NumAnd
        TN = NumNoRec - FN

        if NumAnd == 0:
            # RecallFtem = 0
            Dice = 0
            # SpecifTem = 0
            IoU = 0
        else:
            IoU = NumAnd / (FN + NumRec)
            Dice = 2 * NumAnd / (num_obj + num_pred)
        threshold_Dice[k] = Dice
        threshold_IoU[k] = IoU
    return threshold_Dice, threshold_IoU

if __name__ == '__main__':
    dataset = ['CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB','CVC-300', 'ETIS-LaribPolypDB']
    Dices = []
    IoUs = []
    for data in dataset:
        pred_pth = './runs/test/result_map/'+ data + '/'
        gt_pth = './data/TestDataset/'+data+'/masks/'
        Dice = np.zeros(len(os.listdir(gt_pth)))
        Iou = np.zeros(len(os.listdir(gt_pth)))
        i = 0
        for name in os.listdir(gt_pth):
            pred_mask_pth = pred_pth + name
            pred = cv2.imread(pred_mask_pth, cv2.IMREAD_GRAYSCALE)
            gt_mask_pth = gt_pth+name
            gt = cv2.imread(gt_mask_pth, cv2.IMREAD_GRAYSCALE)
            box_mask = np.zeros_like(gt)
            pred, gt = _prepare_data(pred=pred, gt=gt)
            Dice_list, Iou_list =IOU_dice(pred=pred, gt=gt, thresholds=np.linspace(0.5, 0.5, 1))
            Dice[i] = np.mean(Dice_list)
            Iou[i] = np.mean(Iou_list)
            i+=1
        Dices.append(np.mean(Dice))
        IoUs.append(np.mean(Iou))
        print(data, 'Dice:', np.mean(Dice), 'Iou:', np.mean(Iou))
    mDice = (62*Dices[0] + 100*Dices[1] + 380*Dices[2] + 60*Dices[3] + 196*Dices[4])/(62+100+380+60+196)
    mIoU = (62*IoUs[0] + 100*IoUs[1] + 380*IoUs[2] + 60*IoUs[3] + 196*IoUs[4])/(62+100+380+60+196)
    print('mDice:', mDice, 'mIoU:', mIoU)