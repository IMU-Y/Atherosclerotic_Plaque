from model.unet_parts import *
from functools import partial
nonlinearity = partial(F.relu, inplace=True)

class SE_UNet_CBAM(nn.Module):
    def __init__(self, in_channels=3, output_channels=2, bilinear=True):
        super(SE_UNet_CBAM, self).__init__()
        self.in_channel = in_channels
        self.bilinear = bilinear

        #四次下采样
        self.inc = DoubleConv(in_channels, 64)
        self.inceptionblock0 = Inceptionblock(64)
        self.down1 = Down(64, 128)
        self.inceptionblock1 = Inceptionblock(128)
        self.down2 = Down(128, 256)
        self.inceptionblock2 = Inceptionblock(256)
        self.down3 = Down(256, 512)
        self.inceptionblock3 = Inceptionblock(512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024// factor)

        # self.dblock = DACblock(1024// factor)#提取多尺度信息

        self.spp = SPPLayer(1024 // factor, 3)#金字塔池化层
        self.spp_double_conv = DoubleConv((1024 // factor) * 4, 1024 // factor)
        # self.inceptionblock = Inceptionblock(1024 // factor)  # 提取多尺度信息

        # SE-Module
        # self.se = SELayer((1024 // factor))

        #四次上采样 + CBAM
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.cbam1 = CBAM(512 // factor)

        self.up2 = Up(512, 256 // factor, bilinear)
        self.cbam2 = CBAM(256 // factor)

        self.up3 = Up(256, 128 // factor, bilinear)
        self.cbam3 = CBAM(128 // factor)

        self.up4 = Up(128, output_channels, bilinear)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        # x5 = self(x4).dblock(x4)
        x5 = self.spp(x4)
        x5 = self.spp_double_conv(x5)
        # x5 = self.inceptionblock(x5) #第二版的 SE_UNet，去掉SE_Module,加上 DAC Block

        # x5 = self.se(x5) #第一版的 SE_UNet, spp + SE_Module

        # self.features = []
        x3 = self.inceptionblock3(x3)
        x = self.up1(x5, x3)
        x= self.cbam1(x)

        # self.features.append(x)
        x2 = self.inceptionblock2(x2)
        x = self.up2(x, x2)
        x = self.cbam2(x)

        # self.features.append(x)
        x1 = self.inceptionblock1(x1)
        x = self.up3(x, x1)
        x = self.cbam3(x)
        # self.features.append(x)
        x0 = self.inceptionblock0(x0)
        x = self.up4(x, x0)
        # self.features.append(x)
        return x

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
            tensor = F.interpolate(tensor, size=[h, w])

            tensor_set.append(tensor)
        tensor_set.append(x)
        return torch.cat(tensor_set, dim=1)

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


# 多尺度空洞卷积
class Inceptionblock(nn.Module):
    def __init__(self, channel):
        super(Inceptionblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(self.dilate1(x)))
        dilate3_out = nonlinearity(self.dilate3(self.dilate1(x)))
        dilate4_out = nonlinearity(F.max_pool2d(self.dilate1(x),kernel_size=3,padding=1,stride=1))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out
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