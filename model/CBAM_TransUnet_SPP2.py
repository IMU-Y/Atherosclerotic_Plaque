import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.CBAM import CBAM
from model.unet_parts import DoubleConv
from model.vit import ViT

# 先注意力后spp

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
        self.inc = DoubleConv(in_channels, 64)
        # 192*64
        self.encoder1 = EncoderBottleneck_Trans(64, out_channels * 2)
        # 96*128
        self.encoder2 = EncoderBottleneck1(out_channels * 2, out_channels * 4)
        # 48*256
        self.encoder3 = EncoderBottleneck1(out_channels * 4, out_channels * 8)
        # 24*512
        self.encoder4 = EncoderBottleneck1(out_channels * 8, out_channels * 16)
        self.relu = nn.ReLU(inplace=True)
        self.spp = SPPLayer(out_channels * 16, 3)  # 金字塔池化层
        self.spp_double_conv = DoubleConv(out_channels * 16 * 4, out_channels * 8)
        self.cbam=CBAM(out_channels * 16)


    def forward(self, x):
        x0 = self.inc(x)
        x1,trans_layer = self.encoder1(x0)
        # torch.Size([16, 128, 96, 96])
        x2 = self.encoder2(x1)
        # torch.Size([16, 256, 48, 48])
        x3 = self.encoder3(x2)
        #  torch.Size([16, 512, 24, 24])
        x4 = self.encoder4(x3)
        x5 = x4+self.cbam(x4)
        x5 = self.spp(x5)
        x5 = self.spp_double_conv(x5)
        return x0,x1,x2,trans_layer,x5


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

    def forward(self, x0,x1,x2,trans_layer,x5):
        # 48*48
        # print(x5.shape,print(x3.shape))
        x = self.decoder1(x5, trans_layer)
        # 96*96
        x = self.decoder2(x, x2)
        # 184*184
        x = self.decoder3(x, x1)
        x = self.decoder4(x, x0)

        x = self.out_conv(x)

        return x

class CBAM_TransUnet_SPP2(nn.Module):
    def __init__(self, in_channels, out_put_channels):
        super().__init__()

        self.encoder = Encoder(in_channels)

        self.decoder = Decoder(out_put_channels)

    def forward(self, x):
        x0,x1,x2,trans_layer,x5 = self.encoder(x)
        x = self.decoder(x0,x1,x2,trans_layer,x5)

        return x

