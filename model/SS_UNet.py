from model.unet_parts import *
#计入 se-module 和spp-layer
class SS_UNet(nn.Module):
    def __init__(self, in_channels=3, output_channels=2, bilinear=True):
        super(SS_UNet, self).__init__()
        self.in_channel = in_channels
        self.bilinear = bilinear

        #四次下采样
        channel_list = [-1, 64, 128, 256, 512]

        self.inc = DoubleConv(in_channels, 64)
        self.se1 = SELayer(channel_list[1])
        self.down1 = Down(64, 128)
        self.se2 = SELayer(channel_list[2])
        self.down2 = Down(128, 256)
        self.se3 = SELayer(channel_list[3])
        self.down3 = Down(256, 512)
        self.se4 = SELayer(channel_list[4])
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.spp = SPPLayer(1024 // factor, 3)
        self.spp_double_conv = DoubleConv((1024 // factor) * 4, 1024 // factor)

        #四次上采样
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.out = OutConv(64, output_channels)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x5 = self.spp(x4)
        x5 = self.spp_double_conv(x5)

        x = self.up1(self.se4(x5), self.se4(x3))
        x = self.up2(self.se3(x), self.se3(x2))
        x = self.up3(self.se2(x), self.se2(x1))
        x = self.up4(self.se1(x), self.se1(x0))
        x = self.out(x)
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