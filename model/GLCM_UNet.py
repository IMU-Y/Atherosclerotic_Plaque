from model.unet_parts import *
import numpy as np

class GLCM_UNet(nn.Module):
    def __init__(self, in_channels=3, output_channels=2, bilinear=True):
        super(GLCM_UNet, self).__init__()
        self.in_channel = in_channels
        self.bilinear = bilinear

        #四次下采样
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024// factor)

        self.conv = DoubleConv(1024// factor, 1024 // factor)

        # 四次上采样
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = GLCM_Up(144, output_channels, bilinear)#128+16=144
        self.up4 = Up(128, output_channels, bilinear)

    def forward(self, x,feature):
        x = torch.cat([x, feature], dim=1)
        x0 = self.inc(x)#N*C*H*W
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x5 = self.conv(x4)

        # self.features = []
        temp = self.up1(x5, x3)
        # self.features.append(x)
        temp = self.up2(temp, x2)
        # self.features.append(x)
        temp = self.up3(temp, x1)
        # self.features.append(x)
        temp = self.up4(temp,x0)
        # self.features.append(x)
        return temp

