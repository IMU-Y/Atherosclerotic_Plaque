from model.unet_parts import *

class ASPP_UNet(nn.Module):
    def __init__(self, in_channels=3, output_channels=2, bilinear=True):
        super(ASPP_UNet, self).__init__()
        self.in_channel = in_channels
        self.bilinear = bilinear
        self.norm = nn.BatchNorm2d
        self.conv = nn.Conv2d

        #四次下采样
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024// factor)

        num_classes = output_channels
        self.aspp = ASPP(1024 // factor, (1024 // factor)*4, num_classes, conv=self.conv, norm=self.norm)
        self.aspp_double_conv = DoubleConv((1024 // factor)*4, 1024)

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
        #print(x4.shape)# 512 x 25 x25
        x5 = self.aspp(x4)
        x5 = self.aspp_double_conv(x5)
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

# deeplabv3的ASPP模块
class ASPP(nn.Module):
    def __init__(self, C, depth, num_classes, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()
        self._C = C # 进入aspp的通道数
        self._depth = depth # filter的个数
        self._num_classes = num_classes

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        # 第一个1x1卷积
        self.aspp1 = conv(C, depth, kernel_size=1, stride=1, bias=False)
        # aspp中的空洞卷积，rate=6，12，18
        self.aspp2 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(6*mult), padding=int(6*mult),
                               bias=False)
        self.aspp3 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(12*mult), padding=int(12*mult),
                               bias=False)
        self.aspp4 = conv(C, depth, kernel_size=3, stride=1,
                               dilation=int(18*mult), padding=int(18*mult),
                               bias=False)
        # 对最后一个特征图进行全局平均池化，再feed给256个1x1的卷积核，都带BN
        self.aspp5 = conv(C, depth, kernel_size=1, stride=1, bias=False)

        self.aspp1_bn = norm(depth, momentum)
        self.aspp2_bn = norm(depth, momentum)
        self.aspp3_bn = norm(depth, momentum)
        self.aspp4_bn = norm(depth, momentum)
        self.aspp5_bn = norm(depth, momentum)
        # 先上采样双线性插值得到想要的维度，再进入下面的conv
        self.conv2 = conv(depth * 5, depth, kernel_size=1, stride=1,
                               bias=False)
        self.bn2 = norm(depth, momentum)
        # 打分分类
        self.conv3 = nn.Conv2d(depth, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        # print("input feature size")
        # print(x.shape)
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)

        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)

        # 上采样：双线性插值使x得到想要的维度
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                         align_corners=True)(x5)
        # 经过aspp之后，concat之后通道数变为了5倍
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print("output feature size:")
        # print(x.shape)
        return x