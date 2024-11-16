from model.unet_parts import *

class UNet(nn.Module):
    def __init__(self, in_channels=3, output_channels=2, bilinear=True):
        super(UNet, self).__init__()
        self.in_channel = in_channels
        self.bilinear = bilinear

        #四次下采样
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024// factor)

        self.spp = SPPLayer(1024 // factor, 3)
        self.spp_double_conv = DoubleConv((1024 // factor) * 4, 1024)

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
        # print(x4.shape)
        x5 = self.spp(x4)
        # print(x5.shape)
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