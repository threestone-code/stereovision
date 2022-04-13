"""立体匹配的网络的特征提取子网络"""
import cv2
import torch
import torchvision
from torch import nn
import numpy as np
from torch.nn import functional as F

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}


def conv2d_bn_relu(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU(inplace=True))
def conv2d_bn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


# 基本卷积块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = conv2d_bn_relu(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = conv2d_bn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out
class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        # if self.use_bn:
            # x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x
class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True):
        super(Conv2x, self).__init__()
        self.concat = concat

        if deconv and is_3d:
            kernel = (3, 4, 4)

        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=False, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            self.conv2 = BasicConv(out_channels*2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):

        x = self.conv1(x)
        assert(x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


# the ResNet-like architecture
class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        # 下采样层
        self.conv1 = nn.Sequential(conv2d_bn_relu(1, 32, 3, 2, 1, 1),
                                   conv2d_bn_relu(32, 32, 3, 1, 1, 1),
                                   conv2d_bn_relu(32, 32, 3, 1, 1, 1))
        # 残差层
        self.res_layer1 = self._make_layer(BasicBlock, 32, 32, 1, 1, 1, 1)
        self.res_layer2 = self._make_layer(BasicBlock, 32, 64, 1, 2, 1, 1)
        self.res_layer3 = self._make_layer(BasicBlock, 64, 128, 1, 2, 1, 1)
        self.res_layer4 = self._make_layer(BasicBlock, 128, 128, 1, 1, 1, 1)
        self.reduce = conv2d_bn_relu(128, 32, 3, 1, 1, 1)

        # 沙漏1
        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)
        # 沙漏2（带跳接）
        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride, pad, dilation):
        if stride != 1 or in_planes != out_planes:
            downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, 1, stride, bias=False),
                                       nn.BatchNorm2d(out_planes))
        else:
            downsample = None

        layers = [block(in_planes, out_planes, stride, downsample, pad, dilation)]
        for i in range(1, num_blocks):
            layers.append(block(in_planes, out_planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, X):
        out = self.conv1(X)  # 32,1/2
        # 残差
        out = self.res_layer1(out)  # 32,1/2
        conv0a = out
        out = self.res_layer2(out)  # 64,1/4
        conv1a = out
        out = self.res_layer3(out)  # 128,1/8
        feat0 = out
        out = self.res_layer4(out)  # 128,1/8
        feat1 = out
        out = self.reduce(out)  # 32,1/8
        feat2 = out, rem0 = out
        # 沙漏1
        out = self.conv1a(out)  # 48,1/16
        rem1 = out
        out = self.conv2a(out)  # 64,1/32
        rem2 = out
        out = self.conv3a(out)  # 96,1/64
        rem3 = out
        # 跳接
        out = self.deconv3a(out, rem2)  # 64,1/32
        rem2 = out
        out = self.deconv2a(out, rem1)  # 48,1/16
        rem1 = out
        out = self.deconv1a(out, rem0)  # 32,1/8
        feat3 = out, rem0 = out
        # 沙漏2（带跳接）
        out = self.conv1b(out, rem1)
        rem1 = out
        out = self.conv2b(out, rem2)
        rem2 = out
        out = self.conv3b(out, rem3)
        rem3 = out
        out = self.deconv3b(out, rem2)
        out = self.deconv2b(out, rem1)
        out = self.deconv1b(out, rem0)
        feat4 = out

        # 跨尺度特征图拼接
        feature_map = torch.cat((feat0, feat1, feat2, feat3, feat4), dim=1)
        # 作为引导图的特征图
        guide_feature_map = conv0a
        return guide_feature_map, feature_map


# 特征图可视化
def tensor2img(tensor):
    np_data = tensor.detach().numpy()  # 将tensor数据转为numpy数据
    # normalize，将图像数据扩展到[0,255]
    maxValue = np_data.max()
    np_data = np_data * 255 / maxValue
    mat = np.uint8(np_data)  # float32-->uint8
    # print('mat_shape:', mat.shape)  # mat_shape: (c, h, w)
    mat = mat.transpose(1, 2, 0)  # (c, h, w) ---> (h, w, c)
    return mat
def showimg(mat):
    if mat.shape[2] == 1 or mat.shape[2] == 3:
        cv2.imshow("mat", mat)
    else:
        for i in range(mat.shape[2]):
            cv2.imshow('y' + str(i + 1), mat[:, :, i])
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # img = cv2.imread(r'D:\Users\yzl\Documents\pycharm-workplace\MyProjects\000022_10l.png')
    # trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #                                         torchvision.transforms.Normalize(__imagenet_stats.get('mean'), __imagenet_stats.get('std'))])
    # torch.manual_seed(1)
    # net1 = nn.Sequential(
    #     nn.Conv2d(3, 3, 3, 1, 1),
    # )
    # img = trans(img)
    # y1 = net1(img)
    # y1 = tensor2img(y1)
    # showimg(y1)

    net = feature_extraction()
    X = torch.rand(size=(1, 3, 384, 768))
    # for layer in net:
    #     X = layer(X)
    #     print(layer.__class__.__name__, 'output Size:\t', X.shape)
    # y = net(X)

    print('End.'), print('hello')
