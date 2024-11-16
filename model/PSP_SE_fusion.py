import torch
import torch.nn as nn
from model.unet_parts import DoubleConv
from model.SKNet import SKNet

from model.SE_block import SE_block


class PSP_SE_fusion(nn.Module):
    def __init__(self, model1, model2, in_channels=3, out_put_channels=5, weights=None):
        super(PSP_SE_fusion, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.model1 = model1
        self.model2 = model2
        self.inc = DoubleConv(in_channels, 64)
        self.orin = DoubleConv(64,10)
        self.out_conv1 = nn.Conv2d(64, 5, kernel_size=1)
        self.out_conv2 = nn.Conv2d(64, 5, kernel_size=1)
        self.bott = nn.Sequential(
            nn.Conv2d(20, 64, kernel_size=1),
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.out_conv = nn.Conv2d(64, out_put_channels, kernel_size=1)
        self.se = SE_block(channel=10)

    def forward(self, x):
        # input 是输入图像
        # 使用模型1进行前向传播
        x = self.inc(x)
        seg_map1 = self.model1(x)
        seg_map1 = self.out_conv1(seg_map1)
        ori = self.orin(x)

        # 使用模型2进行前向传播
        seg_map2 = self.model2(x)
        seg_map2 = self.out_conv2(seg_map2)
        sk_result = self.se(torch.cat([seg_map1,seg_map2], dim=1))
        result = torch.cat([ori, sk_result], dim=1)
        result = self.bott(result)
        fused_probs = self.out_conv(result)






        return fused_probs
