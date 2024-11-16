from model.unet_parts import *

class PureUNet(nn.Module):
    def __init__(self, in_channels=3, output_channels=2, bilinear=True):
        super(PureUNet, self).__init__()
        self.in_channel = in_channels
        self.bilinear = bilinear#ConvTranspose2d的效果不如Upsample

        #四次下采样
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
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
        #print(x0.shape) # torch.Size([1, 64, 415, 415])
        x1 = self.down1(x0)
        #print(x1.shape) # torch.Size([1, 128, 207, 207])
        x2 = self.down2(x1)
        #print(x2.shape) # torch.Size([1, 256, 103, 103])
        x3 = self.down3(x2)
        #print(x3.shape) # torch.Size([1, 512, 51, 51])
        x4 = self.down4(x3)
        #print(x4.shape) # torch.Size([1, 512, 25, 25])

        x5 = self.conv(x4)
        #print(x5.shape) # torch.Size([1, 512, 25, 25])
        # self.features = []

        x = self.up1(x5, x3)
        #print(x.shape) # torch.Size([1, 256, 51, 51])
        # self.features.append(x)
        x = self.up2(x, x2)
        #print(x.shape) # torch.Size([1, 128, 103, 103])

        # self.features.append(x)
        x = self.up3(x, x1)
        #print(x.shape) # torch.Size([1, 64, 207, 207])

        # self.features.append(x)
        x = self.up4(x, x0)
        #print(x.shape) # torch.Size([1, 64, 415, 415])

        x = self.out(x)
       #print(x.shape) # torch.Size([1, 4, 415, 415])
        # self.features.append(x)
        # x = nn.Sigmoid()(x)
        return x
