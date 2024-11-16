from model.unet_parts import *
import torch
import torch.nn as nn
import math
from model.unet_parts import *
from torch.nn import functional as F
from model.DA_Attention import DA_Module

##place the Dual-Attention-Module after the decoder part
#特征图太大了，内存溢出
class DA_UNet2(nn.Module):
    def __init__(self, in_channels=3, output_channels=2, bilinear=True):
        super(DA_UNet2, self).__init__()
        self.in_channel = in_channels
        self.bilinear = bilinear

        #四次下采样
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024// factor)

        self.conv = DoubleConv(1024 // factor, 1024)

        #四次上采样
        self.conv1 = DoubleConv(1024, 512)
        self.up1 = Up(1024, 512, bilinear)
        self.conv2 = DoubleConv(512, 256)
        self.up2 = Up(512, 256, bilinear)
        self.conv3 = DoubleConv(256, 128)
        self.up3 = Up(256, 128, bilinear)
        self.conv4 = DoubleConv(128, 64)

        #self.up4 = Up(128, output_channels, bilinear)
        self.up4 = Up(128, output_channels, bilinear)

        #DA-Module
        n_class = output_channels
        self.danet = DA_Module(n_class, output_channels)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        #print(x4.shape)#[1,512,25,25]

        x5 = self.conv(x4)
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
        # DA-Attention
        x = self.danet(x)
        return x