import torch
import torch.nn as nn
from model.unet_parts import DoubleConv
from model.SKNet import SKNet

from model.SE_block import SE_block


class PSP_fusion2(nn.Module):
    def __init__(self, model1, model2, in_channels=3, out_put_channels=5, weights=None):
        super(PSP_fusion2, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.inc = DoubleConv(in_channels, 64)
        self.out_conv1 = nn.Conv2d(64, 32, kernel_size=1)
        self.out_conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.out_conv = nn.Conv2d(128, out_put_channels, kernel_size=1)
        self.se = SE_block(channel=128)

    def forward(self, x):
        # input 是输入图像
        # 使用模型1进行前向传播
        x = self.inc(x)
        seg_map1 = self.model1(x)
        seg_map1 = self.out_conv1(seg_map1)

        # 使用模型2进行前向传播
        seg_map2 = self.model2(x)
        seg_map2 = self.out_conv2(seg_map2)
        result = torch.cat([x, seg_map1, seg_map2], dim=1)
        result = self.se(result)
        fused_probs = self.out_conv(result)






        return fused_probs
