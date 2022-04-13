import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
from . import preprocess
from . import sceneflowlist as lt
from . import readpfm as rp
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return rp.readPFM(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        # 选取图片路径
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]
        # 读取图片
        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL, scaleL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        if self.training:
            # 图片裁剪
            w, h = left_img.size
            # th, tw = 256, 512
            # th, tw = 384, 768
            th, tw = 192, 384
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            dataL = dataL[y1:y1 + th, x1:x1 + tw]
            # # 转换为tensor类型、归一化
            # processed = preprocess.get_transform(augment=False)
            # left_img = processed(left_img)
            # right_img = processed(right_img)
            # return left_img, right_img, dataL
        else:  # 测试使用原图
            pass

        # 转换为tensor类型、归一化
        processed = preprocess.get_transform(augment=False)
        left_img = processed(left_img)
        right_img = processed(right_img)
        return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
