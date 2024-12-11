import torch.nn as nn
import torch
class max_fusion_model(nn.Module):
    def __init__(self, model1, model2, in_channels=3, out_put_channels=5, weights=None):
        super(max_fusion_model, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        # input 是输入图像
        # 使用模型1进行前向传播
        seg_map1 = self.model1(x)

        # 使用模型2进行前向传播
        seg_map2 = self.model2(x)



        # 根据权重加权平均
        fused_probs,_ = torch.max(torch.stack([seg_map1,seg_map2],dim=1),dim=1)




        return fused_probs
