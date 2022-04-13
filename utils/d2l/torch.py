"""
构建自己的d2l
"""
import os

import torch
import random
from torch.utils import data
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from multiprocessing import cpu_count  # 最大同时进程数
# 绘图、展示
import matplotlib_inline
from matplotlib import pyplot as plt
from IPython import display


# 数据加载器

def data_iter(batch_size, features, labels):
    """生成批量数据迭代器"""
    num_features = len(features)
    indexes = list(range(num_features))
    # 随机打乱样本索引
    random.shuffle(indexes)
    for i in range(0, num_features, batch_size):
        batch_indexes = torch.tensor(indexes[i:min(i + batch_size, num_features)])
        yield features[batch_indexes], labels[batch_indexes]


def data_iter_stereo(batch_size, imgls, imgrs, labels):
    """生成批量数据迭代器"""
    num_features = len(imgls)
    indexes = list(range(num_features))
    print(indexes)
    # 随机打乱样本索引
    random.shuffle(indexes)
    for i in range(0, num_features, batch_size):
        batch_indexes = indexes[i:min(i + batch_size, num_features)]
        print(batch_indexes)
        yield imgls[batch_indexes], imgrs[batch_indexes], labels[batch_indexes]


def load_array(data_arrays, batch_size, is_train=True):
    """调用 pytorch 框架 API 构造数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def dataloader(filepath):
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    disp_L = 'disp_occ_0/'
    # disp_R = 'disp_occ_1/'

    image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]

    train = image[:160]
    val = image[160:]

    left_train = [filepath+left_fold+img for img in train]
    right_train = [filepath+right_fold+img for img in train]
    disp_train_L = [filepath+disp_L+img for img in train]
    #disp_train_R = [filepath+disp_R+img for img in train]

    left_val  = [filepath+left_fold+img for img in val]
    right_val = [filepath+right_fold+img for img in val]
    disp_val_L = [filepath+disp_L+img for img in val]
    #disp_val_R = [filepath+disp_R+img for img in val]

    return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L


# 深度学习模型


# 子网结构
class Residual(nn.Module):
    """残差块"""
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            """匹配高、宽、通道数"""
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)



# 损失函数
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def cross_entropy(y_hat, y):
    """交叉熵损失函数"""
    # 使用真实标号 y 作为索引从预测值 y_hat 中取出对应的预测概率
    return -torch.log(y_hat[range(len(y_hat)), y])


# 优化模型
def sgd(params, lr, batch_size):
    """优化算法：小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 分类问题：计算预测是否准确
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        """样本数 > 1 且 类别数 > 1"""
        y_hat = y_hat.argmax(axis=1)  # 返回类别概率最大的索引 == 类别标号
    cmp = y_hat.type(y.dtype) == y  # 统一数据类型，进行比较
    return float(cmp.type(y.dtype).sum()) / len(y)  # 预测正确个数 / 样本数


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 评估模式，只进行forward
    acc = []  # 预测精度
    for X, y in data_iter:
        acc.append(accuracy(net(X), y))  # 计算保存每个批量的精度
    return sum(acc) / len(data_iter)  # 精度求和


# 模型训练脚本
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, nn.Module):
        net.train()
    acc_list = []  # 批量精度列表
    loss_list = []  # 批量损失列表
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)  # 交叉熵
        if isinstance(updater, torch.optim.Optimizer):
            # 框架实现
            updater.zero_grad()
            l.backward()
            updater.step()
        else:  # 自定义实现
            l.sum().backward()
            # 根据批量大小，更新权重
            updater(X.shape[0])
        acc_list.append(accuracy(y_hat, y))
        loss_list.append(float(l.mean()))
    return sum(loss_list) / len(train_iter), sum(acc_list) / len(train_iter)


# 训练函数
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):

    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
    train_loss, train_acc = train_metrics


# 模型测试

# 数据可视化











