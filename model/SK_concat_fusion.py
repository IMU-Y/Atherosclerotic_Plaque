import torch.nn.functional as F
import torch.nn as nn
import torch

# from model.SE_block import SE_block
from model.SKNet import SKNet


class SK_concat_fusion(nn.Module):
    def __init__(self, model1, model2, in_channels=3, out_put_channels=5, weights=None):
        super(SK_concat_fusion, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.sk = SKNet(channel=10)
        self.out = nn.Conv2d(10, out_put_channels, kernel_size=1)

    def forward(self, x):
        # input 是输入图像
        # 使用模型1进行前向传播
        seg_map1 = self.model1(x)

        # 使用模型2进行前向传播
        seg_map2 = self.model2(x)

        x = torch.cat([seg_map1,seg_map2],dim=1)



        fused_probs = self.out(self.sk(x))






        return fused_probs
