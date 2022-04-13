import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
import os
import cv2


class KITTI2015(Dataset):

    def __init__(self, data_dir, mode, validate_size=40, occ=True, transform=None):
        super().__init__()

        self.mode = mode
        self.transform = transform

        if mode == 'train' or mode == 'validate':
            self.dir = os.path.join(data_dir, 'training')
        elif mode == 'test':
            self.dir = os.path.join(data_dir, 'testing')

        left_dir = os.path.join(self.dir, 'image_2')
        right_dir = os.path.join(self.dir, 'image_3')
        left_imgs = list()
        right_imgs = list()

        if mode == 'train':
            imgs_range = range(200 - validate_size)
        elif mode == 'validate':
            imgs_range = range(200 - validate_size, 200)
        elif mode == 'test':
            imgs_range = range(200)

        fmt = '{:06}_10.png'

        for i in imgs_range:
            left_imgs.append(os.path.join(left_dir, fmt.format(i)))
            right_imgs.append(os.path.join(right_dir, fmt.format(i)))

        self.left_imgs = left_imgs
        self.right_imgs = right_imgs

        # self.disp_imgs = None
        if mode == 'train' or mode == 'validate':
            disp_imgs = list()
            if occ:
                disp_dir = os.path.join(self.dir, 'disp_occ_0')
            else:
                disp_dir = os.path.join(self.dir, 'disp_noc_0')
            disp_fmt = '{:06}_10.png'
            for i in imgs_range:
                disp_imgs.append(os.path.join(disp_dir, disp_fmt.format(i)))

            self.disp_imgs = disp_imgs

    def __len__(self):
        return len(self.left_imgs)

    def __getitem__(self, idx):
        data = {}
        # bgr mode
        data['left'] = cv2.imread(self.left_imgs[idx])
        data['right'] = cv2.imread(self.right_imgs[idx])
        if self.mode != 'test':
            data['disp'] = loadPNG16(self.disp_imgs[idx])
        # data['disp'] = cv2.imread(self.left_imgs[idx])[:, :, 0].transpose(2, 0, 1)
        if self.transform:
            # data['left'] = self.transform(data['left'])
            # data['right'] = self.transform(data['right'])
            # print(self.left_imgs[idx])
            data = self.transform(data)
        return data


# 读取视差图
def loadPNG16(file):
    dispmap = cv2.imread(file, flags=cv2.IMREAD_UNCHANGED)
    return dispmap.astype(np.float32) / 256.0


# transforms 数据预处理
class RandomCrop:

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        new_h, new_w = self.output_size
        h, w, _ = sample['left'].shape
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        for key in sample:
            sample[key] = sample[key][top: top + new_h, left: left + new_w]

        return sample


class Normalize:
    '''
    RGB mode
    '''

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['left'] = sample['left'] / 255.0
        sample['right'] = sample['right'] / 255.0

        sample['left'] = self.__normalize(sample['left'])
        sample['right'] = self.__normalize(sample['right'])

        return sample

    def __normalize(self, img):
        for i in range(3):
            img[:, :, i] = (img[:, :, i] - self.mean[i]) / self.std[i]
        return img


class ToTensor:

    def __call__(self, sample):
        left = sample['left']
        right = sample['right']
        disp = sample['disp']
        # H x W x C ---> C x H x W
        sample['left'] = torch.from_numpy(left.transpose([2, 0, 1])).type(torch.FloatTensor)
        sample['right'] = torch.from_numpy(right.transpose([2, 0, 1])).type(torch.FloatTensor)
        sample['disp'] = torch.from_numpy(disp).type(torch.FloatTensor)
        # if 'disp' in sample:
        #     sample['disp'] = torch.from_numpy(sample['disp']).type(torch.FloatTensor)
        return sample


class Pad:
    def __init__(self, H, W):
        self.w = W
        self.h = H

    def __call__(self, sample):
        pad_h = self.h - sample['left'].size(1)
        pad_w = self.w - sample['left'].size(2)

        left = sample['left'].unsqueeze(0)  # [1, 3, H, W]
        left = F.pad(left, pad=(0, pad_w, 0, pad_h))
        right = sample['right'].unsqueeze(0)  # [1, 3, H, W]
        right = F.pad(right, pad=(0, pad_w, 0, pad_h))
        disp = sample['disp'].unsqueeze(0).unsqueeze(1)  # [1, 1, H, W]
        disp = F.pad(disp, pad=(0, pad_w, 0, pad_h))

        sample['left'] = left.squeeze()
        sample['right'] = right.squeeze()
        sample['disp'] = disp.squeeze()

        return sample


import torch
import torchvision.transforms as transforms
import random

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

#__imagenet_stats = {'mean': [0.5, 0.5, 0.5],
#                   'std': [0.5, 0.5, 0.5]}

__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    #if scale_size != input_size:
    #t_list = [transforms.Scale((960,540))] + t_list

    return transforms.Compose(t_list)


def scale_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Scale(scale_size)] + t_list

    transforms.Compose(t_list)


def pad_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    padding = int((scale_size - input_size) / 2)
    return transforms.Compose([
        transforms.RandomCrop(input_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ])


def inception_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomSizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])
def inception_color_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        #transforms.RandomSizedCrop(input_size),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(**normalize)
    ])


def get_transform(name='imagenet', input_size=None,
                  scale_size=None, normalize=None, augment=True):
    normalize = __imagenet_stats
    input_size = 256
    if augment:
            return inception_color_preproccess(input_size, normalize=normalize)
    else:
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)




class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))
