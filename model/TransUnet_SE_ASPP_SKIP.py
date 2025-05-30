import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# from model.CBAM_ASPP import CBAM_ASPP
# from model.SE_block import SE_block
from model.se_ASPP import SE_ASPP
# from model.se_ASPP import SE_ASPP
from model.unet_parts import DoubleConv
from model.vit import ViT
# from model.ASPP_Module import ASPP









class EncoderBottleneck_Trans(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.vit =  ViT(img_dim=384, in_channels=64, embedding_dim=512, head_num=8, mlp_dim=512,
                       block_num=3, patch_dim=8, classification=False, num_classes=5)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        layer=self.vit(x)
        layer = rearrange(layer, "b (x y) c -> b c x y", x=48, y=48)
        return self.maxpool_conv(x),layer




class EncoderBottleneck1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        # self.norm = nn.BatchNorm2d
        # self.conv = nn.Conv2d
        self.inc = DoubleConv(in_channels, 64)
        # 192*128
        self.encoder1 = EncoderBottleneck_Trans(64, out_channels * 2)
        # 96*256
        self.encoder2 = EncoderBottleneck1(out_channels * 2, out_channels * 4)
        # 48*512
        self.encoder3 = EncoderBottleneck1(out_channels * 4, out_channels * 8)
        # 24*1024
        self.encoder4 = EncoderBottleneck1(out_channels * 8, out_channels * 16)
        # self.relu = nn.ReLU(inplace=True)
        # self.aspp = ASPP(out_channels * 8, out_channels * 8*4, 5, conv=self.conv, norm=self.norm)  # 金字塔池化层
        # self.aspp_double_conv = DoubleConv(out_channels * 8*4, out_channels * 16)
        self.conv1 = DoubleConv(out_channels * 16, out_channels * 8)
        self.layer2 = SE_ASPP(out_channels * 4,out_channels * 4)



    def forward(self, x):
        # X0=64*384
        x0 = self.inc(x)
        # X1=128*192,DA_LAYER=512*48
        x1,da_layer4 = self.encoder1(x0)
        # X2=256*96
        x2 = self.encoder2(x1)
        # X3= 512*48
        x3 = self.encoder3(x2)
        # x2 = self.layer2(x2)
        #  torch.Size([16, 512, 24, 24])
        # x4 = self.aspp(x3)
        # x5 = self.aspp_double_conv(x4)
        # x5 = 512*48
        x4 = self.encoder4(x3)
        x2_skip = self.layer2(x2)
        x5 = self.conv1(x4)
        return x0,x1,x2_skip,da_layer4,x5


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

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




class Decoder(nn.Module):
    def __init__(self, out_put_channels):
        super().__init__()

        self.decoder1 = DecoderBottleneck(1024, 256)
        self.decoder2 = DecoderBottleneck(512, 128)
        self.decoder3 = DecoderBottleneck(256, 64)
        self.decoder4 = DecoderBottleneck(128, 64)

        self.out_conv = nn.Conv2d(64, out_put_channels, kernel_size=1)

    def forward(self, x0,x1,x2_skip,da_layer4,x5):
        # 48*48
        # print(x5.shape,print(x3.shape))
        x = self.decoder1(x5, da_layer4)
        # 96*96
        x = self.decoder2(x, x2_skip)
        # 184*184
        x = self.decoder3(x, x1)
        x = self.decoder4(x, x0)

        x = self.out_conv(x)

        return x

class TransUnet_SE_ASPP_SKIP(nn.Module):
    def __init__(self, in_channels, out_put_channels):
        super().__init__()

        self.encoder = Encoder(in_channels)

        self.decoder = Decoder(out_put_channels)

    def forward(self, x):
        x0,x1,x2_skip,da_layer4,x5 = self.encoder(x)
        x = self.decoder(x0,x1,x2_skip,da_layer4,x5)

        return x

