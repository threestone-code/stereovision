import torch
import torch.nn as nn


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(Conv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        return self.relu(self.bn(self.conv2d(X)))


def conv_block(in_channels, out_channels, num, stride, dilation=1):
    layer = [Conv2d(in_channels, out_channels, stride=stride, dilation=dilation)]
    for _ in range(num-1):
        layer.append(Conv2d(out_channels, out_channels, stride=1, dilation=dilation))
    return nn.Sequential(*layer)


class FeatureExtra(nn.Module):
    def __init__(self):
        super(FeatureExtra, self).__init__()
        self.conv1 = conv_block(3, 32, num=9, stride=2)
        self.conv2 = conv_block(32, 64, num=32, stride=2)
        self.conv3 = conv_block(64, 128, num=6, stride=1, dilation=2)
        self.conv4 = conv_block(128, 128, num=6, stride=1, dilation=4)

    def forward(self, X):
        X1 = self.conv2(self.conv1(X))
        X2 = self.conv3(X1)
        X3 = self.conv4(X2)
        return torch.cat((X1, X2, X3), dim=1)


def gwcnet():
    return nn.Sequential(FeatureExtra())


net = gwcnet()
X = torch.rand(size=(12, 3, 256, 256))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'Output Size:\t', X.shape)

# print(net)