import cv2
import os
import numpy as np
import random

from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import transforms as T
from torchvision.transforms import functional as F
 
 
class Resize(object):
    def __init__(self, size):
        self.size = size
 
    def __call__(self, image, box=None):
        image = F.resize(image, self.size)
        box = F.resize(box, self.size, interpolation=Image.BILINEAR)
        return image, box

class adjust_brightness(object):
    def __init__(self, prob):
        self.p = prob
 
    def __call__(self, image, box):
        colorprob = random.uniform(1-self.p, 1+self.p)
        image = F.adjust_brightness(image, colorprob)
        return image, box

class adjust_contrast(object):
    def __init__(self, prob):
        self.p = prob
 
    def __call__(self, image, box):
        colorprob = random.uniform(1-self.p, 1+self.p)
        image = F.adjust_contrast(image, colorprob)
        return image, box

class adjust_hue(object):
    def __init__(self, prob):
        self.p = prob
 
    def __call__(self, image, box):
        colorprob = random.uniform(-self.p, self.p)
        image = F.adjust_hue(image, colorprob)
        return image, box
 
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
 
    def __call__(self, image, box):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, box
 
class ToTensor(object):
    def __call__(self, image, box):
        image = F.to_tensor(image)
        box = F.to_tensor(box)
        return image, box

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
 
    def __call__(self, image, box):
        for t in self.transforms:
            image, box = t(image, box)
        return image, box

def replace_last(string, old, new):
    parts = string.rsplit(old, 1)
    return new.join(parts)


class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, trainsize):
        pic_list = []
        gtbox_list = []
        with open(image_root, 'r') as f:
            for pic in f.readlines():
                pic = pic.strip('\n')
                pic_list.append(pic)
                name = pic.split('/')[-1]
                gtbox_list.append(pic.replace('/images/'+name, '/noisy_labels/sigma_0.2/'+name.replace('.png', '.txt')))

        self.trainsize = trainsize
        self.images = [f for f in pic_list if f.endswith('.jpg') or f.endswith('.png')]
        self.box_file = [f for f in gtbox_list if f.endswith('.txt')]
        self.images = sorted(self.images)
        self.box_file = sorted(self.box_file)
        self.boxeslist = []
        self.imageslist = []
        self.clean_boxeslist = []
        
        for id, box_file in enumerate(self.box_file):
            boxes = []
            clean_boxes = []
            pic = cv2.imread(self.images[id])
            (h,w,c) = pic.shape
            if os.path.getsize(box_file) == 0:
                boxes.append([])
                self.imageslist.append(self.images[id])
                print('txt is empty')
            else:
                with open(box_file, 'r') as f:
                    # with open(self.clean_box_file[id], 'r') as g:
                        for box in f.readlines():
                            box = box.strip('\n')
                            box = box.split(' ')[1:5]
                            # [xmin, ymin, xmax, ymax] = [int(n) for n in box]
                            box = [float(n) for n in box]
                            xmin = int((box[0]-box[2]/2)*w)
                            xmax = int((box[0]+box[2]/2)*w)
                            ymin = int((box[1]-box[3]/2)*h)
                            ymax = int((box[1]+box[3]/2)*h)
                            boxes.append([xmin,ymin,xmax,ymax])
                        self.boxeslist.append(boxes)
                        self.imageslist.append(self.images[id])
        assert(len(self.boxeslist)==len(self.imageslist))
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.transform_aug = Compose([
            Resize((self.trainsize, self.trainsize)),
            adjust_brightness(prob=0.4),
            adjust_contrast(prob=0.4),
            adjust_hue(prob=0.2),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        name = self.imageslist[index]
        image0 = self.rgb_loader(self.imageslist[index])
        boxes = self.boxeslist[index]
        box_mask = np.zeros_like(image0)[:,:,0]
        for box in boxes:
            box_mask[box[1]:box[3], box[0]:box[2]] = 255
        box_mask = Image.fromarray(box_mask)
        image00 = self.gt_transform(image0)
        image = self.img_transform(image0)
        image_aug, box_mask = self.transform_aug(image0, box_mask)
        return image, image_aug, box_mask, image00, name

    def filter_files(self):   #用来验证img和gt_mask的尺寸大小是否一样
        assert len(self.imageslist) == len(self.boxes) == len(self.otherboxes)
        images = []
        gts = []
        for img_path, gt_path in zip(self.imageslist):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(txt_file, batchsize, trainsize, shuffle=True, num_workers=0, pin_memory=True):

    dataset = PolypDataset(txt_file, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class test_dataset_from_txt:
    def __init__(self, image_root, testsize):
        self.testsize = testsize
        self.images = []    
        self.gts =[]
        with open(image_root, 'r') as f:
            for pic in f.readlines():
                pic = pic.strip('\n')
                self.images.append(pic)
                if pic.find('.jpg')>-1:
                    self.gts.append(pic.replace('/images/', '/maskss/').replace('.jpg', '.png'))
                    # gtbox_list.append(pic.replace('images', 'labels').replace('.jpg', '.txt'))
                else:
                    print('image_file is not JPG file')
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        # gt = np.ones((image.size[1], image.size[0]))
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
