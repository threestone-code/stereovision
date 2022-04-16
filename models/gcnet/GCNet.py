import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 3D反卷积时采用卷积进行细化或三线性插值
use_conv3d = True


class ResBlock(nn.Module):
    """2D残差块"""
    def __init__(self, in_planes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        # 残差连接
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ThreeDConv(nn.Module):
    """3D卷积组合"""
    def __init__(self, in_planes, planes, stride=1):
        super(ThreeDConv, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        return out


class GC_NET(nn.Module):
    """2D残差快、3D卷积组合、组件个数、高、宽、最大视差"""
    def __init__(self, block_2d, block_3d, num_block, maxdisp):
        super(GC_NET, self).__init__()
        # self.height = height
        # self.width = width
        self.maxdisp = int(maxdisp/2)

        # 首先对输入进行高宽减半，降低计算量
        self.conv0 = nn.Conv2d(3, 32, 5, 2, 2)
        self.bn0 = nn.BatchNorm2d(32)

        # 8个残差块
        self.res_block_2d = self._make_layer(block_2d, 32, 32, num_block[0], stride=1)

        # last conv2d
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)
        # 这里的左右视图的特征图在通道维拼接形成成本量
        # [B, 2*C, D/2, H/2, W/2]

        # 3D 卷积
        self.conv3d_1 = nn.Conv3d(64, 32, 3, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(32)
        self.conv3d_2 = nn.Conv3d(32, 32, 3, 1, 1)
        self.bn3d_2 = nn.BatchNorm3d(32)

        self.conv3d_3 = nn.Conv3d(64, 64, 3, 2, 1)  # from 21
        self.bn3d_3 = nn.BatchNorm3d(64)
        self.conv3d_4 = nn.Conv3d(64, 64, 3, 2, 1)  # from 24
        self.bn3d_4 = nn.BatchNorm3d(64)
        self.conv3d_5 = nn.Conv3d(64, 64, 3, 2, 1)  # from 27
        self.bn3d_5 = nn.BatchNorm3d(64)

        # conv3d sub_sample block
        self.block_3d_1 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_2 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_3 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_4 = self._make_layer(block_3d, 64, 128, num_block[1], stride=2)

        # deconv3d
        self.deconv1 = nn.ConvTranspose3d(128, 64, 3, 2, 1, 1)
        self.debn1 = nn.BatchNorm3d(64)
        self.deconv2 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn2 = nn.BatchNorm3d(64)
        self.deconv3 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn3 = nn.BatchNorm3d(64)
        self.deconv4 = nn.ConvTranspose3d(64, 32, 3, 2, 1, 1)
        self.debn4 = nn.BatchNorm3d(32)

        # last deconv3d
        self.deconv5 = nn.ConvTranspose3d(32, 1, 3, 2, 1, 1)

        # disparity regression
        self.regression = DisparityRegression(maxdisp)

        if use_conv3d:
            # new add conv3d on the expanding part
            self.upconv3d_1 = nn.Sequential(nn.Conv3d(64, 64, 3, 1, 1), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
            self.upconv3d_2 = nn.Sequential(nn.Conv3d(64, 64, 3, 1, 1), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
            self.upconv3d_3 = nn.Sequential(nn.Conv3d(64, 64, 3, 1, 1), nn.BatchNorm3d(64), nn.ReLU(inplace=True))
            self.upconv3d_4 = nn.Sequential(nn.Conv3d(32, 32, 3, 1, 1), nn.BatchNorm3d(32), nn.ReLU(inplace=True))

    def forward(self, imgLeft, imgRight):
        original_size = [1, self.maxdisp*2, imgLeft.size(2), imgLeft.size(3)]
        print('input shape: ', imgLeft.shape)
        imgl0 = F.relu(self.bn0(self.conv0(imgLeft)))  # [B 32 H/2 W/2]
        imgr0 = F.relu(self.bn0(self.conv0(imgRight)))

        imgl_block = self.res_block_2d(imgl0)  # [B 32 H/2 W/2]
        imgr_block = self.res_block_2d(imgr0)

        imgl1 = self.conv1(imgl_block)  # [B 32 H/2 W/2]
        imgr1 = self.conv1(imgr_block)

        # cost volume
        cost_volume = self.cost_volume(imgl1, imgr1)  # [B, 2*C, D/2, H/2, W/2]
        print('cost shape: ', cost_volume.shape)

        conv3d_out = F.relu(self.bn3d_1(self.conv3d_1(cost_volume)))
        conv3d_out = F.relu(self.bn3d_2(self.conv3d_2(conv3d_out)))  # [B 32 D/2 H/2 W/2]
        print('layer1 shape: ', conv3d_out.shape)

        # conv3d block
        conv3d_block_1 = self.block_3d_1(cost_volume)  # [B 64 D/4 H/4 W/4]
        print('layer2 shape: ', conv3d_block_1.shape)
        conv3d_21 = F.relu(self.bn3d_3(self.conv3d_3(cost_volume)))

        conv3d_block_2 = self.block_3d_2(conv3d_21)  # [B 64 D/8 H/8 W/8]
        print('layer3 shape: ', conv3d_block_2.shape)

        conv3d_24 = F.relu(self.bn3d_4(self.conv3d_4(conv3d_21)))

        conv3d_block_3 = self.block_3d_3(conv3d_24)  # [B 64 D/16 H/16 W/16]
        print('layer4 shape: ', conv3d_block_3.shape)

        conv3d_27 = F.relu(self.bn3d_5(self.conv3d_5(conv3d_24)))

        conv3d_block_4 = self.block_3d_4(conv3d_27)  # [B 128 D/32 H/32 W/32]
        print('layer5 shape: ', conv3d_block_4.shape)


        if use_conv3d:
            # deconv
            deconv3d = F.relu(self.debn1(self.deconv1(conv3d_block_4)) + conv3d_block_3)
            deconv3d = self.upconv3d_1(deconv3d)
            print('de layer4 shape: ', deconv3d.shape)
            deconv3d = F.relu(self.debn2(self.deconv2(deconv3d)) + conv3d_block_2)
            deconv3d = self.upconv3d_2(deconv3d)
            print('de layer3 shape: ', deconv3d.shape)

            deconv3d = F.relu(self.debn3(self.deconv3(deconv3d)) + conv3d_block_1)
            deconv3d = self.upconv3d_3(deconv3d)
            print('de layer2 shape: ', deconv3d.shape)

            deconv3d = F.relu(self.debn4(self.deconv4(deconv3d)) + conv3d_out)
            deconv3d = self.upconv3d_4(deconv3d)
            print('de layer1 shape: ', deconv3d.shape)

        else:
            # deconv
            deconv3d = F.relu(self.debn1(self.deconv1(conv3d_block_4)) + conv3d_block_3)
            deconv3d = F.relu(self.debn2(self.deconv2(deconv3d)) + conv3d_block_2)
            deconv3d = F.relu(self.debn3(self.deconv3(deconv3d)) + conv3d_block_1)
            deconv3d = F.relu(self.debn4(self.deconv4(deconv3d)) + conv3d_out)

        # last deconv3d
        deconv3d = self.deconv5(deconv3d)
        print('last layer shape: ', deconv3d.shape)

        out = deconv3d.view(original_size)  # [1, 96*2, H, W]
        prob = F.softmax(-out, 1)  # 求视差概率

        # 回归得到视差图 [B, H, W]
        disp1 = self.regression(prob)
        return disp1

    def _make_layer(self, block_2d, in_planes, planes, num_block, stride):
        strides = [stride]+[1]*(num_block-1)
        layers = []
        for s in strides:
            layers.append(block_2d(in_planes, planes, s))
        return nn.Sequential(*layers)

    def cost_volume(self, imgl, imgr):
        """扩充特征图到指定视差范围，然后拼接得到匹配代价空间"""
        B, C, H, W = imgl.size()
        cost_vol = torch.zeros(B, C * 2, self.maxdisp, H, W).type_as(imgl)
        for i in range(self.maxdisp):
            if i > 0:
                cost_vol[:, :C, i, :, i:] = imgl[:, :, :, i:]
                cost_vol[:, C:, i, :, i:] = imgr[:, :, :, :-i]
            else:
                cost_vol[:, :C, i, :, :] = imgl
                cost_vol[:, C:, i, :, :] = imgr
        return cost_vol.contiguous()

    def __init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


def GcNet(maxdisp):
    return GC_NET(ResBlock, ThreeDConv, [8, 1], maxdisp)


class DisparityRegression(nn.Module):

    def __init__(self, max_disp):
        super().__init__()

        self.disp_score = torch.arange(0, max_disp)  # [D]
        self.disp_score = self.disp_score.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, D, 1, 1]

    def forward(self, prob):
        disp_score = self.disp_score.expand_as(prob).type_as(prob)  # [B, D, H, W]
        # 视差加权求和
        out = torch.sum(disp_score * prob, dim=1)  # [B, H, W]
        return out


if __name__ == '__main__':
    net = GcNet(256, 256, 192)
    left = torch.rand(size=(1, 3, 64, 64))
    right = torch.rand(size=(1, 3, 64, 64))
    disp = net(left, right)
