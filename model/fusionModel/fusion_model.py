import torch.nn as nn
import torch
class fusion_model(nn.Module):
    def __init__(self, model1, model2, in_channels=3, out_put_channels=5, weights=None):
        super(fusion_model, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.weights = weights if weights is not None else [0.55, 0.45]

    def forward(self, x):
        # input 是输入图像
        # 使用模型1进行前向传播
        seg_map1 = self.model1(x)

        # 使用模型2进行前向传播
        seg_map2 = self.model2(x)



        # 根据权重加权平均
        fused_probs = self.weights[0] * seg_map1 + self.weights[1] * seg_map2




        return fused_probs
