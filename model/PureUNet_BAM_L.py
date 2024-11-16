from model.unet_parts import *
from model.BAM import *

class PureUNet_BAM_L(nn.Module):
    def __init__(self, in_channels=3, output_channels=2, bilinear=True):
        super(PureUNet_BAM_L, self).__init__()
        self.in_channel = in_channels
        self.bilinear = bilinear#ConvTranspose2d的效果不如Upsample

        # 四次下采样
        # 在编码器的尾部添加BAM
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.bam3 = BAM(512)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024// factor)
        self.conv = DoubleConv(1024 // factor, 1024 // factor)

        #四次上采样
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.out = OutConv(64,output_channels)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.bam3(self.down3(x2))
        x4 = self.down4(x3)

        x5 = self.conv(x4)
        # self.features = []

        x = self.up1(x5, x3)
        # print(x.shape)
        # self.features.append(x)
        x = self.up2(x, x2)
        # self.features.append(x)
        x = self.up3(x, x1)
        # self.features.append(x)
        x = self.up4(x, x0)

        x = self.out(x)
        # self.features.append(x)
        # x = nn.Sigmoid()(x)
        return x