import torch
import torch.nn as nn
from torch.nn import functional as F


input_size = (192, 384)


def conv2d(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU())


def convTranspose2d(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU())


def loss(pre, gt):
    return torch.sqrt(torch.mean(torch.square(pre - gt)))


class DispNetS(nn.Module):
    """FCN网络"""
    def __init__(self):
        super(DispNetS, self).__init__()
        # the extraction partS
        # 左右帧堆叠输入，in_channels=6
        self.conv1 = conv2d(6, 64, 7, 2, 3)
        self.conv2 = conv2d(64, 128, 5, 2, 2)
        self.conv3a = conv2d(128, 256, 5, 2, 2)
        self.conv3b = conv2d(256, 256, 3, 1, 1)
        self.conv4a = conv2d(256, 512, 3, 2, 1)
        self.conv4b = conv2d(512, 512, 3, 1, 1)
        self.conv5a = conv2d(512, 512, 3, 2, 1)
        self.conv5b = conv2d(512, 512, 3, 1, 1)
        self.conv6a = conv2d(512, 1024, 3, 2, 1)
        self.conv6b = conv2d(1024, 1024, 3, 1, 1)

        # the expanding part
        self.upconv5 = convTranspose2d(1024, 512, 4, 2, 1)
        self.catpr6 = convTranspose2d(1, 1, 4, 2, 1)
        self.iconv5 = conv2d(1025, 512, 3, 1, 1)

        self.upconv4 = convTranspose2d(512, 256, 4, 2, 1)
        self.catpr5 = convTranspose2d(1, 1, 4, 2, 1)
        self.iconv4 = conv2d(769, 256, 3, 1, 1)

        self.upconv3 = convTranspose2d(256, 128, 4, 2, 1)
        self.catpr4 = convTranspose2d(1, 1, 4, 2, 1)
        self.iconv3 = conv2d(385, 128, 3, 1, 1)

        self.upconv2 = convTranspose2d(128, 64, 4, 2, 1)
        self.catpr3 = convTranspose2d(1, 1, 4, 2, 1)
        self.iconv2 = conv2d(193, 64, 3, 1, 1)

        self.upconv1 = convTranspose2d(64, 32, 4, 2, 1)
        self.catpr2 = convTranspose2d(1, 1, 4, 2, 1)
        self.iconv1 = conv2d(97, 32, 3, 1, 1)

        # predict results
        self.pr6out = nn.Conv2d(1024, 1, 3, 1, 1)
        self.pr5out = nn.Conv2d(512, 1, 3, 1, 1)
        self.pr4out = nn.Conv2d(256, 1, 3, 1, 1)
        self.pr3out = nn.Conv2d(128, 1, 3, 1, 1)
        self.pr2out = nn.Conv2d(64, 1, 3, 1, 1)
        self.pr1out = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, imgl, imgr):
        self.outSize = (imgl.shape[2], imgl.shape[3])
        # 图像对堆叠输入
        X = torch.cat((imgl, imgr), dim=1)
        # the contraction part
        X = self.conv1(X)
        conv1 = X
        X = self.conv2(X)
        conv2 = X
        X = self.conv3a(X)
        X = self.conv3b(X)
        conv3b = X
        X = self.conv4a(X)
        X = self.conv4b(X)
        conv4b = X
        X = self.conv5a(X)
        X = self.conv5b(X)
        conv5b = X
        X = self.conv6a(X)
        X = self.conv6b(X)
        conv6b = X

        # the expanding part and output
        self.pr6 = self.pr6out(conv6b)

        X = self.upconv5(X)
        X = self.iconv5(torch.cat((X, self.catpr6(self.pr6), conv5b), dim=1))
        self.pr5 = self.pr5out(X)

        X = self.upconv4(X)
        X = self.iconv4(torch.cat((X, self.catpr5(self.pr5), conv4b), dim=1))
        self.pr4 = self.pr4out(X)

        X = self.upconv3(X)
        X = self.iconv3(torch.cat((X, self.catpr4(self.pr4), conv3b), dim=1))
        self.pr3 = self.pr3out(X)

        X = self.upconv2(X)
        X = self.iconv2(torch.cat((X, self.catpr3(self.pr3), conv2), dim=1))
        self.pr2 = self.pr2out(X)

        X = self.upconv1(X)
        X = self.iconv1(torch.cat((X, self.catpr2(self.pr2), conv1), dim=1))
        self.pr1 = self.pr1out(X)

        out = self.pr1
        # return out, self.pr2, self.pr3, self.pr4, self.pr5, self.pr6

        # 为了计算多尺度loss，多尺度训练
        # 将pre1-6上采样到dispL相同维度，再计算loss
        # 三种上采样方法：转置卷积、插值、反池化，这里选双线性插值
        if self.training:
            self.pr1 = F.interpolate(self.pr1, self.outSize, mode='bilinear')
            self.pr2 = F.interpolate(self.pr2, self.outSize, mode='bilinear')
            self.pr3 = F.interpolate(self.pr3, self.outSize, mode='bilinear')
            self.pr4 = F.interpolate(self.pr4, self.outSize, mode='bilinear')
            self.pr5 = F.interpolate(self.pr5, self.outSize, mode='bilinear')
            self.pr6 = F.interpolate(self.pr6, self.outSize, mode='bilinear')
            return self.pr1, self.pr2, self.pr3, self.pr4, self.pr5, self.pr6
        else:
            self.pr1 = F.interpolate(self.pr1, self.outSize, mode='bilinear')
            return self.pr1


# --------------------------------------------------------------------
def build_c_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, i, :, i:] = correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, i, :, :] = correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    # print('volume shape: ', volume.shape)
    return volume


def correlation(fea1, fea2):
    B, C, H, W = fea1.shape
    cost = (fea1 * fea2).view([B, C, H, W]).mean(dim=1)
    assert cost.shape == (B, H, W)
    return cost


class DispNetC(nn.Module):
    """FCN网络"""
    def __init__(self, maxdisp):
        super(DispNetC, self).__init__()
        self.maximum_disp = maxdisp
        # the extraction partS
        self.conv1 = conv2d(3, 64, 7, 2, 3)
        self.conv2 = conv2d(64, 128, 5, 2, 2)
        self.conv3a = conv2d(self.maximum_disp // 4, 256, 5, 2, 2)
        self.conv3b = conv2d(256, 256, 3, 1, 1)
        self.conv4a = conv2d(256, 512, 3, 2, 1)
        self.conv4b = conv2d(512, 512, 3, 1, 1)
        self.conv5a = conv2d(512, 512, 3, 2, 1)
        self.conv5b = conv2d(512, 512, 3, 1, 1)
        self.conv6a = conv2d(512, 1024, 3, 2, 1)
        self.conv6b = conv2d(1024, 1024, 3, 1, 1)

        # the expanding part
        self.upconv5 = convTranspose2d(1024, 512, 4, 2, 1)
        self.catpr6 = convTranspose2d(1, 1, 4, 2, 1)
        self.iconv5 = conv2d(1025, 512, 3, 1, 1)

        self.upconv4 = convTranspose2d(512, 256, 4, 2, 1)
        self.catpr5 = convTranspose2d(1, 1, 4, 2, 1)
        self.iconv4 = conv2d(769, 256, 3, 1, 1)

        self.upconv3 = convTranspose2d(256, 128, 4, 2, 1)
        self.catpr4 = convTranspose2d(1, 1, 4, 2, 1)
        self.iconv3 = conv2d(385, 128, 3, 1, 1)

        self.upconv2 = convTranspose2d(128, 64, 4, 2, 1)
        self.catpr3 = convTranspose2d(1, 1, 4, 2, 1)
        self.iconv2 = conv2d(64+1+128, 64, 3, 1, 1)

        self.upconv1 = convTranspose2d(64, 32, 4, 2, 1)
        self.catpr2 = convTranspose2d(1, 1, 4, 2, 1)
        self.iconv1 = conv2d(32+1+64, 32, 3, 1, 1)

        # predict results
        self.pr6out = conv2d(1024, 1, 3, 1, 1)
        self.pr5out = conv2d(512, 1, 3, 1, 1)
        self.pr4out = conv2d(256, 1, 3, 1, 1)
        self.pr3out = conv2d(128, 1, 3, 1, 1)
        self.pr2out = conv2d(64, 1, 3, 1, 1)
        self.pr1out = conv2d(32, 1, 3, 1, 1)

    def forward(self, imgl, imgr):
        # 左右图像分别处理
        imgl1 = self.conv1(imgl)
        imgl2 = self.conv2(imgl1)

        imgr1 = self.conv1(imgr)
        imgr2 = self.conv2(imgr1)

        # 计算特征图的水平相关
        cost = build_c_volume(imgl2, imgr2, self.maximum_disp // 4)

        X = self.conv3a(cost)
        X = self.conv3b(X)
        conv3b = X
        X = self.conv4a(X)
        X = self.conv4b(X)
        conv4b = X
        X = self.conv5a(X)
        X = self.conv5b(X)
        conv5b = X
        X = self.conv6a(X)
        X = self.conv6b(X)
        conv6b = X

        # the expanding part and output
        self.pr6 = self.pr6out(conv6b)

        X = self.upconv5(X)
        X = self.iconv5(torch.cat((X, self.catpr6(self.pr6), conv5b), dim=1))
        self.pr5 = self.pr5out(X)

        X = self.upconv4(X)
        X = self.iconv4(torch.cat((X, self.catpr5(self.pr5), conv4b), dim=1))
        self.pr4 = self.pr4out(X)

        X = self.upconv3(X)
        X = self.iconv3(torch.cat((X, self.catpr4(self.pr4), conv3b), dim=1))
        self.pr3 = self.pr3out(X)

        X = self.upconv2(X)
        X = self.iconv2(torch.cat((X, self.catpr3(self.pr3), imgl2), dim=1))
        self.pr2 = self.pr2out(X)

        X = self.upconv1(X)
        X = self.iconv1(torch.cat((X, self.catpr2(self.pr2), imgl1), dim=1))
        self.pr1 = self.pr1out(X)

        out = self.pr1
        # 为了计算多尺度loss，多尺度训练
        # 将pre1-6上采样到dispL相同维度，再计算loss
        # 三种上采样方法：转置卷积、插值、反池化，这里选双线性插值
        if self.training:
            self.pr1 = F.interpolate(self.pr1, input_size, mode='bilinear')
            self.pr2 = F.interpolate(self.pr2, input_size, mode='bilinear')
            self.pr3 = F.interpolate(self.pr3, input_size, mode='bilinear')
            self.pr4 = F.interpolate(self.pr4, input_size, mode='bilinear')
            self.pr5 = F.interpolate(self.pr5, input_size, mode='bilinear')
            self.pr6 = F.interpolate(self.pr6, input_size, mode='bilinear')
            return self.pr1, self.pr2, self.pr3, self.pr4, self.pr5, self.pr6
        else:
            self.pr1 = F.interpolate(self.pr1, input_size, mode='bilinear')
            return out
