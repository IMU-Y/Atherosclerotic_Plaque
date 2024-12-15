import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv
        )
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # 调整维度顺序以适应Mamba
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        x = self.mamba(x)
        x = self.norm(x)
        # 恢复原始维度
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return x

class VSSBlock(nn.Module):
    """Visual Selective Scan Block - 修正版
    每个VSS Block只包含一个MambaBlock
    """
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        # 只包含一个MambaBlock
        self.mamba = MambaBlock(d_model, d_state, d_conv)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        identity = x
        
        # 通过单个MambaBlock
        x = self.mamba(x)
        
        # 残差连接
        x = x + identity
        
        # Layer Normalization
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        
        return x

class PatchMerging(nn.Module):
    """Patch Merging Layer"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduction = nn.Linear(4 * in_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(4 * in_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], 1)
        
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.reduction(x)
        x = x.permute(0, 3, 1, 2)
        
        return x

class PatchExpanding(nn.Module):
    """Patch Expanding Layer"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.expand = nn.Linear(in_channels, 2 * 2 * out_channels, bias=False)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.expand(x)
        x = x.reshape(b, h, w, 2, 2, -1)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(b, -1, h*2, w*2)
        
        return x

class MambaUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dims=[64, 128, 256, 512]):
        super().__init__()
        
        # 初始卷积
        self.conv_first = nn.Conv2d(in_channels, dims[0], kernel_size=7, padding=3)
        
        # 编码器
        self.encoder_stages = nn.ModuleList()
        for i in range(len(dims)-1):
            stage = nn.Sequential(
                VSSBlock(dims[i]),
                VSSBlock(dims[i]),
                PatchMerging(dims[i], dims[i+1])
            )
            self.encoder_stages.append(stage)
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            VSSBlock(dims[-1]),
            VSSBlock(dims[-1])
        )
        
        # 解码器
        self.decoder_stages = nn.ModuleList()
        for i in range(len(dims)-1, 0, -1):
            stage = nn.Sequential(
                PatchExpanding(dims[i], dims[i-1]),
                VSSBlock(dims[i-1]),
                VSSBlock(dims[i-1]),
            )
            self.decoder_stages.append(stage)
            
        # 输出层
        self.final_conv = nn.Conv2d(dims[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        # 初始特征提取
        x = self.conv_first(x)
        
        # 编码器路径
        encoder_features = []
        for stage in self.encoder_stages:
            encoder_features.append(x)
            x = stage(x)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 解码器路径
        for i, stage in enumerate(self.decoder_stages):
            # 上采样和特征融合
            x = stage[0](x)  # PatchExpanding
            x = x + encoder_features[-(i+1)]  # 跳跃连接
            x = stage[1:](x)  # VSS Blocks
        
        # 输出层
        x = self.final_conv(x)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)