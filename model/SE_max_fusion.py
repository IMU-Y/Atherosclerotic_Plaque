import torch
import torch.nn as nn

from model.SE_block import SE_block


class SE_max_fusion(nn.Module):
    def __init__(self, model1, model2, in_channels=3, out_put_channels=5, weights=None):
        super(SE_max_fusion, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.se = SE_block(out_put_channels*2)
        self.out_conv = nn.Conv2d(out_put_channels*2, out_put_channels, kernel_size=1)

    def forward(self, x):
        # input 是输入图像
        # 使用模型1进行前向传播
        seg_map1 = self.model1(x)

        # 使用模型2进行前向传播
        seg_map2 = self.model2(x)
        result = torch.cat([seg_map1, seg_map2], dim=1)
        result = self.se(result)
        result1, result2 = torch.split(result, split_size_or_sections=5, dim=1)

        fused_probs, _ = torch.max(torch.stack([result1, result2], dim=1), dim=1)






        return fused_probs
