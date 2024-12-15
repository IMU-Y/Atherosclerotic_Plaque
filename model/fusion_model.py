import torch.nn as nn
import torch
class fusion_model(nn.Module):
    def __init__(self, mamba_model, trans_model, in_channels=3, out_put_channels=5):
        super(fusion_model, self).__init__()
        self.mamba_model = mamba_model
        self.trans_model = trans_model
        print('max_fusion_model')
        
    def get_mamba_parameters(self):
        """获取MambaUNet部分的参数"""
        return self.mamba_model.parameters()
        
    def get_trans_parameters(self):
        """获取TransUNet部分的参数"""
        return self.trans_model.parameters()

    def forward(self, x):
        mamba_out = self.mamba_model(x)
        trans_out = self.trans_model(x)
        fused_probs, _ = torch.max(torch.stack([mamba_out, trans_out], dim=1), dim=1)
        return fused_probs
