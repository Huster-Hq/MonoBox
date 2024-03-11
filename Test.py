import torch
import torch.nn.functional as F
import numpy as np
import os, argparse, imageio

from lib.polyp_pvt_tsn import PVT_TSN
from utils.dataloader import test_dataset
from skimage import img_as_ubyte


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='/data/hq/code/polyp/image_segmentation/IBoxCLA/results/experiment_rank_loss/try/Publicdataset_sigma0.2/IBox_rank_v1/snapshots/24PVT_CPD_best.pth')
opt = parser.parse_args()
model = PVT_TSN()


model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(opt.pth_path).items()})
model.cuda()
model.eval()

for _data_name in ['CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB','CVC-300', 'ETIS-LaribPolypDB']:
    data_path = './data/TestDataset/{}'.format(_data_name)
    save_path = './runs/test/result_map/{}/'.format(_data_name)
    if not os.path.exists(save_path):
       os.makedirs(save_path, exist_ok=True)

    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        # P1, _,_ = model(image, step=4, cur_epoch = i+5)   #step=3:student prediction, step=4, teacher prediction
        P1,P2,_,_ = model(image, step=4, cur_epoch = i+5)   #step=3:student prediction, step=4, teacher prediction
        res = F.upsample(P1+P2, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imsave(save_path+name, img_as_ubyte(res))
