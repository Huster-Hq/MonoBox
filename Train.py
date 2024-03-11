import torch
import os
import numpy as np
import argparse
import torch.nn.functional as F
import logging

from torch.autograd import Variable
from datetime import datetime
from lib.polyp_pvt_tsn import PVT_TSN
from lib.losses import MC_loss, structure_loss
from utils.fromtxt_dataloader_box_supervised import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter, mask2boxes, merge_box
from torch.utils.tensorboard import SummaryWriter



def test(model, path, dataset):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC_T, DSC_S = 0.0, 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        model = model.cuda()
        lateral_map_1_T,lateral_map_2_T,_,_ = model(image, step=4, cur_epoch = i+5)
        lateral_map_1_S,lateral_map_2_S,_,_ = model(image, step=3, cur_epoch = i+5)
        # eval Teacher Dice
        res_T = F.upsample((lateral_map_1_T+lateral_map_2_T) , size=gt.shape, mode='bilinear', align_corners=False)
        res_T = res_T.sigmoid().data.cpu().numpy().squeeze()
        res_T = (res_T - res_T.min()) / (res_T.max() - res_T.min() + 1e-8)
        input_T = res_T
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat_T = np.reshape(input_T, (-1))
        target_flat = np.reshape(target, (-1))
        intersection_T = (input_flat_T * target_flat)
        dice_T = (2 * intersection_T.sum() + smooth) / (input_T.sum() + target.sum() + smooth)
        dice_T = '{:.4f}'.format(dice_T)
        dice_T = float(dice_T)
        DSC_T = DSC_T + dice_T
        # eval Student Dice
        res_S = F.upsample((lateral_map_1_S+lateral_map_2_S) , size=gt.shape, mode='bilinear', align_corners=False)
        res_S = res_S.sigmoid().data.cpu().numpy().squeeze()
        res_S = (res_S - res_S.min()) / (res_S.max() - res_S.min() + 1e-8)
        input_S = res_S
        input_flat_S = np.reshape(input_S, (-1))
        intersection_S = (input_flat_S * target_flat)
        dice_S = (2 * intersection_S.sum() + smooth) / (input_S.sum() + target.sum() + smooth)
        dice_S = '{:.4f}'.format(dice_S)
        dice_S = float(dice_S)
        DSC_S = DSC_S + dice_S

    return DSC_T / num1, DSC_S / num1



def train(train_loader, model, optimizer, epoch, writer, confuse_rate):
    global best
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_IBox_record, loss_CLA_record, loss_Pixel_record, loss_Total_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    # ---- correct unconfident scale ----
    if epoch%10==1 and epoch!=1:
        confuse_rate = confuse_rate/2
    for id, pack in enumerate(train_loader, start=1):
        if epoch>=1:
            for t,rate in enumerate(size_rates):
                optimizer.zero_grad()
                # ---- data prepare ----
                images, images_aug, box_mask, img0, name = pack
                images = Variable(images).cuda()
                images_aug = Variable(images_aug).cuda()
                box_mask = Variable(box_mask).cuda()
                with torch.no_grad():
                    lateral_map_1_T, lateral_map_2_T, FP_T, BP_T = model(images, step=2, cur_epoch=epoch, kernel=True, boxmask=box_mask)
                    mask_T = torch.sigmoid(lateral_map_1_T + lateral_map_2_T)
                
                # ---- rescale ----
                trainsize = int(round(opt.trainsize*rate/32)*32)
                kernelsize = int(trainsize/8)
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    images_aug = F.upsample(images_aug, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    mask_T = F.upsample(mask_T, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    box_mask = F.upsample(box_mask, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    BP_T = F.upsample(BP_T, size=(kernelsize, kernelsize), mode='bilinear', align_corners=True)
                box_mask[box_mask>=0.5] = 1
                box_mask[box_mask<0.5] = 0

                # ---- initial noisy boxes ----
                if epoch == 1:
                    box_mask2box = mask2boxes(box_mask)
                    for i in range(len(box_mask2box)):
                        id_box_dict[name[i]+str(t)] = box_mask2box[i]
                
                # ---- using Label Correction every 10 epochs ----
                if epoch%10==1 and epoch!=1:
                    box_mask_ori = torch.zeros_like(box_mask)
                    for index, img_name in enumerate(name):
                        box_ori = id_box_dict[img_name+str(t)]
                        for bbox in box_ori:
                            [xmin, ymin, xmax, ymax] = [element for element in bbox]
                            box_mask_ori[index,:,ymin:ymax+1,xmin:xmax+1] = 1
                    box_mask_cor = merge_box(box_mask_ori, mask_T, iou_thres=0.5)
                    box_mask_cor2box = mask2boxes(box_mask_cor)
                    for i in range(len(box_mask_cor)):
                        id_box_dict[name[i]+str(t)] = box_mask_cor2box[i]
                box_mask_x = torch.zeros_like(box_mask)
                for index, img_name in enumerate(name):
                    box_x = id_box_dict[img_name+str(t)]
                    for bbox in box_x:
                        [xmin, ymin, xmax, ymax] = [element for element in bbox]
                        box_mask_x[index,:,ymin:ymax+1,xmin:xmax+1] = 1
                
                # ---- inference ----
                P1, P2, sim = model(images_aug, step = 1, cur_epoch = epoch, kernel=False, FP=FP_T, BP=BP_T)               
                
                # ---- training loss ----
                loss_IBox1 = MC_loss(torch.sigmoid(P1), box_mask_x, confuse_rate)
                loss_IBox2 = MC_loss(torch.sigmoid(P2), box_mask_x, confuse_rate)
                loss_CLA = MC_loss(sim[:,1,:,:].unsqueeze(1), box_mask_x, confuse_rate)
                loss_Pixel1 = structure_loss(P1, (mask_T.detach()*(box_mask_x)))
                loss_Pixel2 = structure_loss(P2, (mask_T.detach()*(box_mask_x)))
                loss = loss_IBox1 + loss_IBox2 + loss_CLA + 0.5*loss_Pixel1 + 0.5*loss_Pixel2

                writer.add_scalar("loss_IBox",loss_IBox1.item(), (epoch-1)*total_step+id)
                writer.add_scalar("loss_CLA",loss_CLA.item(), (epoch-1)*total_step+id)
                writer.add_scalar("loss_Pixel",loss_Pixel2.item(), (epoch-1)*total_step+id)
                writer.add_scalar("loss_Total",loss.item(), (epoch-1)*total_step+id)

                # ---- backward ----
                loss.backward()
                clip_gradient(optimizer, opt.clip)
                optimizer.step()
                # ---- recording loss ----
                if rate == 1:
                    loss_IBox_record.update(loss_IBox1.data, opt.batchsize)
                    loss_CLA_record.update(loss_CLA.data, opt.batchsize)
                    loss_Pixel_record.update(loss_Pixel1.data, opt.batchsize)
                    loss_Total_record.update(loss.data, opt.batchsize)
            # ---- train visualization ----
            if id % 20 == 0 or id == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                    '[loss_Total: {:.4f}, loss_IBox: {:.4f}, loss_CLA: {:0.4f}, loss_Pixel:{:0.4f}'.
                    format(datetime.now(), epoch, opt.epoch, id, total_step,
                            loss_Total_record.show(), loss_IBox_record.show(), loss_CLA_record.show(), loss_Pixel_record.show()))

    # ---- eval and save model ----
    save_path = './{}/weigths/'.format(opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    test1path = './data/TestDataset'
    if (epoch + 1) % 1 == 0:
        meandice_T, meandice_S = 0, 0
        weight = [62, 100, 380, 60, 196]
        for index, dataset in enumerate(['CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'CVC-300', 'ETIS-LaribPolypDB']):
            dataset_dice_T, dataset_dice_S = test(model, test1path, dataset)
            logging.info('epoch: {}, dataset: {}, Teadice: {}, Studice:{}'.format(epoch, dataset, dataset_dice_T, dataset_dice_S))
            print(dataset, ': ', dataset_dice_T, dataset_dice_S)
            meandice_T += dataset_dice_T * weight[index]
            meandice_S += dataset_dice_S * weight[index]
        meandice = meandice_S if meandice_S>meandice_T else meandice_T
        meandice = meandice / (62+100+380+60+196)
        print('mean-dice:{}'.format(meandice))
        logging.info('####epoch: {}, meandice:{}'.format(epoch, meandice))
        torch.save(model.state_dict(), save_path +str(epoch)+ 'PVT_CPD.pth')
        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), save_path +str(epoch)+ 'PVT_CPD_best.pth')
            print('##############################################################################best', best)
            logging.info('##############################################################################best:{}'.format(best))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=24, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')
    parser.add_argument('--unconf_scale', type=float,
                    default=0.2, help='uncofident_scale')
    parser.add_argument('--train_path', type=str,
                        default='./data/TrainDataset/CVC550+Kvasir', help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='runs/MonoBox1')
    opt = parser.parse_args()

    if not os.path.exists(opt.train_save):
        os.makedirs(opt.train_save)

    logging.basicConfig(filename=opt.train_save+'/MonoBox-IBoxCLA-Sythetic.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    # ---- build models ----
    # torch.cuda.set_device([0,1])  # set your gpu device
    model = PVT_TSN().cuda()

    params = model.parameters()
    # optimizer = torch.optim.Adam(params, opt.lr)
    optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    writer = SummaryWriter('snapshots/{}/'.format(opt.train_save))

    print("*"*20, "Start Loader", "*"*20)
    train_loader = get_loader('./data/train.txt', batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)
    best = 0

    print("#"*20, "Start Training", "#"*20)
    
    global id_box_dict
    id_box_dict = {}
    
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, 200)
        train(train_loader, model, optimizer, epoch, writer, opt.unconf_scale)
