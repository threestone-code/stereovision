import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    """DER：下采样、特征提取、残差块"""
    def __init__(self, input_channels, out_channels, use_1x1conv=True, strides=2):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            """匹配高、宽、通道数"""
            self.conv3 = nn.Conv2d(input_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


if __name__ == '__main__':
    net = nn.Conv2d(3, 3, 1, 1, 0, bias=False)
    X = torch.rand(size=(1, 3, 3, 3), dtype=torch.float32)
    net.weight.data.fill_(1.0)
    y = net(X)
    print(X)
    print(net.weight.data)
    print(y)
