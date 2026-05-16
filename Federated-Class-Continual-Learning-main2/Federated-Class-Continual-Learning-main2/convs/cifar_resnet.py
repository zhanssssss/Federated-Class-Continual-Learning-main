'''
Reference:
https://github.com/khurramjaved96/incremental-learning/blob/autoencoders/model/resnet32.py
'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from convs.linears import SimpleLinear


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class DownsampleB(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleB, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DownsampleC(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nIn != nOut
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsampleD(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleD, self).__init__()
        assert stride == 2
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=2, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNetBasicblock(nn.Module):
    """
    ResNet的基本块，包含两个3x3的卷积层，以及一个残差连接。

    参数:
    - inplanes: 输入通道数
    - planes: 输出通道数
    - stride: 卷积步长，默认为1
    - downsample: 下采样层，用于匹配输入输出维度，默认为None

    属性:
    - expansion: 扩展倍数，本基本块未使用扩展，故为1
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        """
        前向传播函数。

        参数:
        - x: 输入特征数据

        返回:
        - 激活函数后的输出特征数据
        """
        residual = x  # 保存输入，用于残差连接

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)  # 第一个卷积层处理，加上激活函数

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)  # 第二个卷积层处理，不改变维度

        if self.downsample is not None:
            residual = self.downsample(x)  # 如果有下采样层，对输入进行下采样，以匹配输出维度

        return F.relu(residual + basicblock, inplace=True)  # 将残差与卷积层输出相加，并通过激活函数


class CifarResNet(nn.Module):
    """
    CifarResNet模型的构造函数。

    参数：
    - block: 构建残差块的类型。
    - depth: 模型的深度，用于确定网络中卷积层的数量。
    - channels: 输入图像的通道数，默认为3（RGB图像）。

    返回值：
    - 无
    """

    def __init__(self, block, depth, channels=3):
        super(CifarResNet, self).__init__()

        # 确保模型深度适合CIFAR数据集，即深度必须是6的倍数加2
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        # 初始化网络的卷积层和批量归一化层
        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        # 构建网络的三个阶段，每个阶段都是由若干个残差块组成
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.out_dim = 64 * block.expansion
        self.fc = nn.Linear(64 * block.expansion, 10)
        self.proj_head = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_dim, 128)
        )  # 定义一个投影头部，包括两个全连接层和一个ReLU激活函数
        # 使用特定的初始化方法对模型的参数进行初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        # block: 残差块的类，例如BasicBlock或Bottleneck
        # planes: 输出通道数
        # blocks: 在当前层中重复残差块的次数
        # stride: 卷积层的步长，当stride大于1时，会导致空间维度减小

        downsample = None  # 初始化下采样模块为None
        # 如果步长不为1，或者输入通道数与输出通道数不一致时，
        # 需要一个下采样层来调整维度，以便于残差连接后可以相加
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []  # 初始化包含残差块的列表
        # 添加第一个残差块，并且可能包含下采样
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion  # 更新输入通道数以匹配下一个块的维度
        # 对剩余的块进行迭代，它们的步长为1，并且不需要下采样
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        # 使用nn.Sequential将所有的残差块组合为一个连续的模块，并返回
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)  # [bs, 16, 32, 32]
        x = F.relu(self.bn_1(x), inplace=True)

        x_1 = self.stage_1(x)  # [bs, 16, 32, 32]
        x_2 = self.stage_2(x_1)  # [bs, 32, 16, 16]
        x_3 = self.stage_3(x_2)  # [bs, 64, 8, 8]

        pooled = self.avgpool(x_3)  # [bs, 64, 1, 1]
        features = pooled.view(pooled.size(0), -1)  # [bs, 64]
        proj = F.normalize(self.proj_head(features), dim=1)

        return {
            'fmaps': [x_1, x_2, x_3],
            'features': features,
            'proj': proj
        }

    @property
    def last_conv(self):
        return self.stage_3[-1].conv_b


class SupConMlp(nn.Module):
    """
    用于构建监督对比学习任务的ResNet编码器及其投影头。
    参数:
    - name: 使用的ResNet变体名称，默认为'resnet32'。
    - head: 投影头的类型，可以选择'linear'或'mlp'，默认为'mlp'。
    - feat_dim: 投影头输出特征维度，默认为128。
    """

    def __init__(self, feat_dim=128):
        super(SupConMlp, self).__init__()
        # 初始化ResNet编码器和输入维度
        dim_in = 64
        # 根据head参数选择不同的投影头
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim)
        )

    def reinit_head(self):
        """
        重置投影头的参数，通常在训练开始或迁移学习时使用。
        """
        for layers in self.head.children():
            if hasattr(layers, 'reset_parameters'):
                layers.reset_parameters()

    def forward(self, x, return_feat=False, norm=True):
        """
        定义网络的前向传播路径。

        参数:
        - x: 输入的图像张量。
        - return_feat: 是否返回编码器的中间特征，默认为False，只返回投影头的输出。
        - norm: 是否对投影头的输出进行标准化，默认为True。

        返回:
        - 如果return_feat为False，返回投影头的输出；
        - 如果return_feat为True，返回投影头的输出和编码器的中间特征。
        """
        # 根据norm参数决定是否对特征进行标准化
        if norm:
            feat = F.normalize(self.head(x), dim=1)
        else:
            feat = self.head(x)
        # 根据return_feat参数决定返回的内容
        if return_feat:
            return feat, x
        else:
            return feat


def resnet20mnist():
    """Constructs a ResNet-20 model for MNIST."""
    model = CifarResNet(ResNetBasicblock, 20, 1)
    return model


def resnet32mnist():
    """Constructs a ResNet-32 model for MNIST."""
    model = CifarResNet(ResNetBasicblock, 32, 1)
    return model


def resnet20():
    """Constructs a ResNet-20 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 20)
    return model


def resnet32():
    """Constructs a ResNet-32 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 32)
    return model


def resnet44():
    """Constructs a ResNet-44 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 44)
    return model


def resnet56():
    """Constructs a ResNet-56 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 56)
    return model


def resnet110():
    """Constructs a ResNet-110 model for CIFAR-10."""
    model = CifarResNet(ResNetBasicblock, 110)
    return model
