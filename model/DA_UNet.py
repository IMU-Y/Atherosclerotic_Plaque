import torch
import torch.nn as nn
import math
from model.unet_parts import *
from torch.nn import functional as F
from model.DA_Attention import DA_Module

#在bottle neck 融合DANet的注意力机制
class DA_UNet(nn.Module):
    def __init__(self, in_channels=3, output_channels=2, bilinear=True):
        super(DA_UNet, self).__init__()
        self.in_channel = in_channels
        self.bilinear = bilinear

        #四次下采样
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024// factor)

        # danet模块替换  1024// factor == 512
        n_class = output_channels
        self.danet = DA_Module(n_class, 1024// factor)
        self.danet_double_conv = DoubleConv(1024 // factor, 1024)


        #四次上采样
        self.conv1 = DoubleConv(1024, 512)
        self.up1 = Up(1024, 512, bilinear)
        self.conv2 = DoubleConv(512, 256)
        self.up2 = Up(512, 256, bilinear)
        self.conv3 = DoubleConv(256, 128)
        self.up3 = Up(256, 128, bilinear)
        self.conv4 = DoubleConv(128, 64)
        self.up4 = Up(128, output_channels, bilinear)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        #print(x4.shape)#[1,512,25,25]
        #DA-Attention
        x5 = self.danet(x4)
        x5 = self.danet_double_conv(x5)

        # print('spp_double_conv shape:', x5.shape)
        self.features = []
        x5 = self.conv1(x5)
        x = self.up1(x5, x3)
        # print(x.shape)
        self.features.append(x)
        x = self.conv2(x)
        x = self.up2(x, x2)
        # print(x.shape)
        self.features.append(x)
        x = self.conv3(x)
        x = self.up3(x, x1)
        # print(x.shape)
        self.features.append(x)
        x = self.conv4(x)
        x = self.up4(x, x0)

        self.features.append(x)
        # x = nn.Sigmoid()(x)
        return x