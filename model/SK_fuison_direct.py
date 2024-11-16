from model.SKmodule import SKmodule
from model.unet_parts import DoubleConv
import torch.nn.functional as F
import torch.nn as nn
import torch


# class EncoderBottleneck(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )
#
#     def forward(self, x):
#         return self.maxpool_conv(x)
#
#
# class DecoderBottleneck(nn.Module):
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super().__init__()
#         if bilinear:
#             self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = nn.Sequential(
#                 nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True)
#             )
#         else:
#             self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)
#
#     def forward(self, x1, x2):
#         x = self.upsample(x1)
#         x = self.conv(x)
#         # print(x.shape)
#         if x2 is not None:
#             diffY = x2.size()[2] - x.size()[2]
#             diffX = x2.size()[3] - x.size()[3]
#             x = F.pad(x, [diffX // 2, diffX - diffX // 2,
#                           diffY // 2, diffY - diffY // 2])
#             x = torch.cat([x2, x], dim=1)
#             # print(x.shape)
#         return x
#
#
# class Encoder(nn.Module):
#     def __init__(self, out_channels=5):
#         super().__init__()
#         # 192*10
#         self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2)
#         # 96*20
#         self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4)
#         # 48*40
#         self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8)
#
#
#     def forward(self, x0):
#         # x1=192*10
#         x1 = self.encoder1(x0)
#         # x2=96*20
#         x2 = self.encoder2(x1)
#         # x3=48*40
#         x3 = self.encoder3(x2)
#         return x0, x1, x2, x3
#
# class Decoder(nn.Module):
#     def __init__(self, out_channels=5):
#         super().__init__()
#         self.conv1 = DoubleConv(40,20)
#         self.conv2 = DoubleConv(20,10)
#         self.conv3 = DoubleConv(10,5)
#
#         self.decoder1 = DecoderBottleneck(40, 20)
#         self.decoder2 = DecoderBottleneck(20, 10)
#         self.decoder3 = DecoderBottleneck(10, 5)
#
#     def forward(self,x0,x1,x2,v):
#         # 48*48
#         x = self.decoder1(v, x2)
#         x = self.conv1(x)
#         # 96*96
#         x = self.decoder2(x, x1)
#         x = self.conv2(x)
#         # 192*192
#         x = self.decoder3(x, x0)
#         x = self.conv3(x)
#
#
#         return x

class SK_fusion_direct(nn.Module):
    def __init__(self, model1, model2, in_channels=3, out_put_channels=5, weights=None):
        super(SK_fusion_direct, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.sk = SKmodule(channel=5)
        # self.encoder1 = Encoder(out_channels=5)
        # self.encoder2 = Encoder(out_channels=5)
        # self.decoder1 = Decoder(out_channels=5)
        # self.decoder2 = Decoder(out_channels=5)
    def forward(self, x):
        # input 是输入图像
        # 使用模型1进行前向传播
        seg_map1 = self.model1(x)

        # 使用模型2进行前向传播
        seg_map2 = self.model2(x)
        # x10,x11,x12,v1 = self.encoder1(seg_map1)
        # x20,x21,x22,v2 = self.encoder2(seg_map2)
        r1,r2 = self.sk(seg_map1,seg_map2)
        # result1 = self.decoder1(x10,x11,x12,r1)
        # result2 = self.decoder2(x20,x21,x22,r2)




        # 根据权重加权平均
        fused_probs = r1+r2




        return fused_probs
