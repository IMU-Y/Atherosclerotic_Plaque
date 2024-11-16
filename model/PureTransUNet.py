import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from model.unet_parts import DoubleConv

from model.vit import ViT


class EncoderBottleneck0(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(out_channels)
        self.relu0 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        return x


class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)

        return x





class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels=32):
        super().__init__()

        # 192*64
        self.encoder1 = EncoderBottleneck0(in_channels, out_channels * 2)
        # 96*128
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        # 48*256
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)
        # 24*512
        self.encoder4 = EncoderBottleneck(out_channels * 8, out_channels * 16, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.vit_img_dim = img_dim // 8
        # B*576*512
        self.vit = ViT(img_dim=self.vit_img_dim, in_channels=256, embedding_dim=512, head_num=8, mlp_dim=512,
                       block_num=1, patch_dim=2, classification=False, num_classes=5)

        self.conv2 = nn.Conv2d(512, 256, kernel_size=1, stride=1)
        self.norm2 = nn.BatchNorm2d(256)

    def forward(self, x):
        # torch.Size([16, 64, 192, 192])
        x1 = self.encoder1(x)
        # torch.Size([16, 128, 96, 96])
        x2 = self.encoder2(x1)
        # torch.Size([16, 256, 48, 48])
        x3 = self.encoder3(x2)
        #  torch.Size([16, 512, 24, 24])
        x4 = self.encoder4(x3)
        # 得到位置注意力特征图
        #  torch.Size([16, 512, 24, 24])
        temp = self.vit(x3)
        #  torch.Size([16, 512, 24, 24])
        temp = rearrange(temp, "b (x y) c -> b c x y", x=self.vit_img_dim // 2, y=self.vit_img_dim // 2)
        x5 = temp
        x5 = self.conv2(x5)
        x5 = self.norm2(x5)
        # torch.Size([16, 256, 24, 24])
        x5 = self.relu(x5)
        # print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)

        return x1, x2, x3, x5


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        # if bilinear:
        #     self.upsample = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        #     self.layer = nn.Sequential(
        #         nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1),
        #         nn.BatchNorm2d(in_channels // 2),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, stride=1, padding=1),
        #         nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True)
        #     )
        # else:
        #     self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        #     self.layer = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        #         nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        #         nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True)
        #     )

    def forward(self, x1, x2):
        x = self.upsample(x1)
        # print(x.shape)
        if x2 is not None:
            diffY = x2.size()[2] - x.size()[2]
            diffX = x2.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x], dim=1)
            # print(x.shape)
        x = self.conv(x)
        return x


class OneUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.upsample1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2=None):
        x = self.upsample1(x1)
        if x2 is not None:
            x = torch.cat([x2, x], dim=1)

        x = self.conv(x)
        return x



class Decoder(nn.Module):
    def __init__(self, out_put_channels):
        super().__init__()

        self.decoder1 = DecoderBottleneck(512, 128)
        self.decoder2 = DecoderBottleneck(256, 64)
        self.decoder3 = DecoderBottleneck(128, 32)
        self.decoder4 = OneUp(32, 16)

        self.out_conv = nn.Conv2d(16, out_put_channels, kernel_size=1)

    def forward(self, x1, x2, x3, x5):
        # 48*48
        # print(x5.shape,print(x3.shape))
        x = self.decoder1(x5, x3)
        # 96*96
        x = self.decoder2(x, x2)
        # 184*184
        x = self.decoder3(x, x1)
        x = self.decoder4(x)

        x = self.out_conv(x)

        return x

class PureTransUNet(nn.Module):
    def __init__(self, img_dim, in_channels, out_put_channels):
        super().__init__()

        self.encoder = Encoder(img_dim, in_channels)

        self.decoder = Decoder(out_put_channels)

    def forward(self, x):
        x1, x2, x3, x5 = self.encoder(x)
        x = self.decoder(x1, x2, x3, x5)

        return x