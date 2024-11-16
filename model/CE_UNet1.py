from model.unet_parts import *
from functools import partial
nonlinearity = partial(F.relu, inplace=True)

class CE_UNet(nn.Module):
    def __init__(self, in_channels=3, output_channels=2, bilinear=True):
        super(CE_UNet, self).__init__()
        self.in_channel = in_channels
        self.bilinear = bilinear

        #四次下采样
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024// factor)

        self.dblock = DACblock(1024// factor)
        self.spp = SPPblock(1024// factor)
        self.spp_double_conv = DoubleConv(516, 1024)

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

        x5 = self.dblock(x4)
        x5 = self.spp(x5)
        x5 = self.spp_double_conv(x5)
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

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        # self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        # self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        # self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        # self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        self.layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear')
        self.layer2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear')
        self.layer3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear')
        self.layer4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear')

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out