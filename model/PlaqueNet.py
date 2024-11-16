import torch
import math
from model.unet_parts import *
from model.CBAM import CBAM

class PlaqueNet(nn.Module):
    def __init__(self, in_channels=3, output_channels=2, bilinear=True):
        super(PlaqueNet, self).__init__()
        self.in_channel = in_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.res_se1 = Residual_SE(1)
        self.down1 = Down(64, 128)
        self.res_se2 = Residual_SE(2)
        self.down2 = Down(128, 256)
        self.res_se3 = Residual_SE(3)
        self.down3 = Down(256, 512)
        self.res_se4 = Residual_SE(4)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.spp = SPPLayer(1024 // factor, 3)
        self.spp_double_conv = DoubleConv((1024 // factor) * 4, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.cbam1 = CBAM(512 // factor)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.cbam2 = CBAM(256 // factor)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.cbam3 = CBAM(128 // factor)
        self.up4 = Up(128, output_channels, bilinear)

    def forward(self, x):

        x1 = self.res_se1(self.inc(x))
        # print(x1.shape)
        x2 = self.res_se2(self.down1(x1))
        # print(x2.shape)
        x3 = self.res_se3(self.down2(x2))
        # print(x3.shape)
        x4 = self.res_se4(self.down3(x3))
        # print(x4.shape)
        x5 = self.down4(x4)
        x5 = self.spp(x5)

        # print("===================")
        x5 = self.spp_double_conv(x5)
        # print('spp_double_conv shape:', x5.shape)
        self.features = []
        x = self.up1(x5, x4)
        # print(x.shape)
        x = self.cbam1(x)
        # print(x.shape)
        self.features.append(x)
        x = self.up2(x, x3)
        x = self.cbam2(x)
        # print(x.shape)
        self.features.append(x)
        x = self.up3(x, x2)
        x = self.cbam3(x)
        # print(x.shape)
        self.features.append(x)
        x = self.up4(x, x1)
        # print(x.shape)
        self.features.append(x)
        return x

class Squeuzed_Block(nn.Module):

    def __init__(self, layer_num, output_channel):
        super(Squeuzed_Block, self).__init__()

        in_features_list = [-1, output_channel * 370 * 370, output_channel * 185 * 185, output_channel * 92 * 92, output_channel * 46 * 46]
        kernel_size_list = [-1, 10, 5, 2, 1]
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size_list[layer_num])
        self.by_pass = nn.Sequential(
            nn.Linear(in_features=in_features_list[layer_num] // (kernel_size_list[layer_num] ** 2), out_features=in_features_list[layer_num] // (kernel_size_list[layer_num] ** 2 * output_channel)),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=in_features_list[layer_num] // (kernel_size_list[layer_num] ** 2 * output_channel), out_features=output_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        pool_x = self.avg_pool(x)
        pool_x = pool_x.view(pool_x.shape[0], -1)
        by_pass = self.by_pass(pool_x)
        by_pass = by_pass.unsqueeze(dim=-1).unsqueeze(dim=-1)
        # print('by_pass.shape', by_pass.shape)
        return x * by_pass


class Residual_SE(nn.Module):
    def __init__(self, layer_num):
        super(Residual_SE, self).__init__()
        channel_list = [-1, 64, 128, 256, 512]
        # channel_list = [-1, 32, 64, 128, 256]
        # channel_list = [-1, 16, 32, 64, 128]
        in_channels = channel_list[layer_num]
        self.left = nn.Sequential(
            DoubleConv(in_channels, in_channels),
            # Squeuzed_Block(layer_num, in_channels),
            SELayer(in_channels),
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.right = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)
        return torch.cat([left, right], dim=1)

# 构建SPP层(空间金字塔池化层)
class SPPLayer(torch.nn.Module):

    def __init__(self, in_channels, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.static_kernel_size = [1, 2, 4]
        self.num_levels = num_levels
        self.pool_type = pool_type

        self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1)

    def forward(self, x):
        num, c, h, w = x.size()     # num:样本数量 c:通道数 h:高 w:宽
        tensor_set = []
        for i in range(self.num_levels):
            level = i+1
            kernel_size = self.static_kernel_size[i]
            # kernel_size = (math.ceil(h / level), math.ceil(w / level))
            # print('kernel size:', kernel_size)
            # stride = (math.ceil(h / level), math.ceil(w / level))
            # pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                # tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling)
                tensor = F.max_pool2d(x, kernel_size=kernel_size)
            else:
                # tensor = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling)
                tensor = F.avg_pool2d(x, kernel_size=kernel_size)

            # conv and upsample
            tensor = F.relu(self.conv(tensor))
            tensor = F.upsample_bilinear(tensor, size=[h, w])

            tensor_set.append(tensor)
        tensor_set.append(x)
        return torch.cat(tensor_set, dim=1)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)