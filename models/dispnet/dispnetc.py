import torch
from torch import nn


def conv2d(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True))


def convTranspose2d(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True))


# def conv3d(in_channels, out_channels, kernel_size, stride, padding):
#     return nn.Sequential(
#         nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
#         nn.BatchNorm3d(out_channels),
#         nn.LeakyReLU(inplace=True))


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
        self.iconv2 = conv2d(64+1+128*2, 64, 3, 1, 1)

        self.upconv1 = convTranspose2d(64, 32, 4, 2, 1)
        self.catpr2 = convTranspose2d(1, 1, 4, 2, 1)
        self.iconv1 = conv2d(32+1+64*2, 32, 3, 1, 1)

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
        X = self.iconv2(torch.cat((X, self.catpr3(self.pr3), imgl2, imgr2), dim=1))
        self.pr2 = self.pr2out(X)

        X = self.upconv1(X)
        X = self.iconv1(torch.cat((X, self.catpr2(self.pr2), imgl1, imgr1), dim=1))
        self.pr1 = self.pr1out(X)

        out = self.pr1
        return out, self.pr2, self.pr3, self.pr4, self.pr5, self.pr6


if __name__ == '__main__':
    net = DispNetC(192)
    X = torch.rand(1, 3, 768, 384)
    y = net(X, X)
    for i in range(len(y)):
        print('pre' + str(i + 1) + ' shape:\t', y[i].shape)
    print('End.')


# def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
#     B, C, H, W = refimg_fea.shape
#     volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
#     for i in range(maxdisp):
#         if i > 0:
#             volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
#             volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
#         else:
#             volume[:, :C, i, :, :] = refimg_fea
#             volume[:, C:, i, :, :] = targetimg_fea
#     volume = volume.contiguous()
#     return volume


def build_c_volume(refimg_fea, targetimg_fea, maxdisp, num_groups=1):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i], num_groups=1)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups=1)
    volume = volume.contiguous()
    return volume


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups=1):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    # 这里分组相关和全相关一样？
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost














