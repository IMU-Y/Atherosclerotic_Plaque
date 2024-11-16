import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    # 如果conv标志为True，转置数组维度（可能将HWIO转换为OIHW）
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    # 将NumPy数组转换为PyTorch张量
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        # 获取卷积核的权重参数
        w = self.weight
        # 计算卷积核权重参数的方差（v）和均值（m），对权重参数的每个通道进行计算
        # 对于卷积核权重参数的维度顺序，O（输出通道数）通常是第0维，I（输入通道数）通常是第1维，H（卷积核的高度）通常是第2维，W（卷积核的宽度）通常是第3维
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        # 标准化卷积核的权重参数：(w - m) / sqrt(v + 1e-5)
        w = (w - m) / torch.sqrt(v + 1e-5)
        # 使用标准化后的卷积核进行正常的卷积操作
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


# 创建一个3x3的卷积层，使用自定义的 StdConv2d 类。
#
#     参数:
#     - cin: 输入通道数（Input Channels）。
#     - cout: 输出通道数（Output Channels）。
#     - stride: 卷积的步幅（stride），默认值为1。
#     - groups: 卷积的分组数（groups），默认值为1。
#     - bias: 是否使用偏置项，默认为False。
#
#     返回:
#     - 一个使用 StdConv2d 类创建的卷积层。
def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)

    # 参数:
    # - cin: 输入通道数（Input Channels）。
    # - cout: 输出通道数（Output Channels）。
    # - stride: 卷积的步幅（stride），默认值为1。
    # - bias: 是否使用偏置项，默认为False。
    #
    # 返回:
    # - 一个使用 StdConv2d 类创建的卷积层。
def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


# 预激活（Pre-activation）的瓶颈块实现
class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        # 如果未提供cout或cmid的值，会使用默认值。cout默认为cin，cmid默认为cout的四分之一。
        cout = cout or cin
        cmid = cmid or cout//4
        # 第一层：Group Normalization + 1x1卷积
        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        # 第二层：GroupNormalization + 3x3卷积（stride可能不为1）
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        # 第三层：Group Normalization + 1x1卷积
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)

        # 如果步幅不为1或输入通道数不等于输出通道数，则进行投影（downsample）
        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            # 这行代码创建了一个conv1x1类型的投影层self.downsample。该投影层使用1x1的卷积核，调整输入通道数cin到输出通道数cout，并可以设置步幅stride。该投影用于使残差分支的维度匹配单元的输出，以确保二者可以相加。
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            # 用于对投影后的残差进行归一化。这是为了保持网络的稳定性和加速训练的技巧。
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        # 如果存在 downsample 属性，说明需要进行投影
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        # 第一层：1x1卷积 -> Group Normalization -> ReLU激活
        y = self.relu(self.gn1(self.conv1(x)))
        # 第二层：3x3卷积 -> Group Normalization -> ReLU激活
        y = self.relu(self.gn2(self.conv2(y)))
        # 第三层：1x1卷积 -> Group Normalization
        y = self.gn3(self.conv3(y))
        # 将残差和单元的输出相加，再通过 ReLU 激活
        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        # 加载卷积层的权重
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        # 加载 Group Normalization 的权重和偏置
        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        # 将加载的权重复制到相应的模块中
        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        # 如果存在 downsample 属性，表示有投影层
        if hasattr(self, 'downsample'):
            # 加载投影层的权重
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            # 将加载的权重复制到相应的投影层中
            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        # 根据给定的 width_factor 计算网络的宽度
        width = int(64 * width_factor)
        self.width = width

        # 根层,包括 7x7 的卷积、Group Normalization 和 ReLU 激活。注释中有一个被注释掉的最大池化层。
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        # 主体层
        self.body = nn.Sequential(OrderedDict([
            # 第一个块，包含了若干预激活瓶颈块，其中的第一个瓶颈块的输入通道 cin 为 width，输出通道 cout 为 width*4，cmid 为 width。
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            # 第二个块，包含了若干预激活瓶颈块，其中的第一个瓶颈块的输入通道 cin 为 width*4，输出通道 cout 为 width*8，cmid 为 width*2，且带有步幅 stride=2。
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            # 第三个块，包含了若干预激活瓶颈块，其中的第一个瓶颈块的输入通道 cin 为 width*8，输出通道 cout 为 width*16，cmid 为 width*4，且带有步幅 stride=2。
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        # 用于存储每个阶段的特征
        features = []
        # 获取输入尺寸信息
        b, c, in_size, _ = x.size()
        # 根层的前向传播
        x = self.root(x)
        features.append(x)
        # 第一个池化层
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        # 主体层的前向传播
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            # 调整特征尺寸
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            # 将调整后的特征添加到列表中
            features.append(feat)
        # 最后一个块的前向传播
        x = self.body[-1](x)
        # 返回最终输出和特征列表（反转顺序）
        return x, features[::-1]
