import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation块"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)

class MambaBlock(nn.Module):
    """增强版MambaBlock"""
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv
        )
        self.norm = nn.LayerNorm(d_model)
        # 添加SE注意力
        self.se = SEBlock(d_model)
        # 可选：添加投影层确保残差连接维度匹配
        self.proj = nn.Conv2d(d_model, d_model, 1) if d_model != d_model else nn.Identity()
        
    def forward(self, x):
        # 保存输入用于残差连接
        identity = x
        
        # Mamba处理
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        x = self.mamba(x)
        x = self.norm(x)
        
        # 恢复维度
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        # 应用SE注意力
        x = self.se(x)
        
        # 残差连接
        x = x + self.proj(identity)
        
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.mamba = MambaBlock(out_channels)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.mamba(x)
        skip = x
        p = self.pool(x)
        return skip, p

class ChannelEmbeddings(nn.Module):
    """通道嵌入模块"""
    def __init__(self, in_channels, patch_size):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=patch_size,
            stride=patch_size,
            padding=patch_size//2  # 添加padding以保持尺寸
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.norm = nn.LayerNorm(in_channels)
        
    def forward(self, x):
        if x is None:
            return None
        # 保存输入尺寸
        _, _, h, w = x.shape
        x = self.patch_embeddings(x)
        # 确保输出尺寸与输入相同
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        x = x + self.position_embeddings
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x

class MambaChannelTransformer(nn.Module):
    """Mamba版本的通道变压器"""
    def __init__(self, channels=[64, 128, 256, 512]):
        super().__init__()
        # 通道嵌入
        self.embeddings = nn.ModuleList([
            ChannelEmbeddings(c, patch_size=2) for c in channels
        ])
        
        # Mamba编码器
        self.encoders = nn.ModuleList([
            MambaBlock(c) for c in channels
        ])
        
        # 特征重建
        self.reconstructs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True)
            ) for c in channels
        ])

    def forward(self, features):
        outputs = []
        for i, feat in enumerate(features):
            if feat is not None:
                # 通道嵌入
                x = self.embeddings[i](feat)
                # Mamba编码
                x = self.encoders[i](x)
                # 特征重建
                x = self.reconstructs[i](x)
                # 残差连接
                x = x + feat
                outputs.append(x)
            else:
                outputs.append(None)
        return outputs

class UpBlock(nn.Module):
    """增强版上采样块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # 通道交叉注意力模块
        self.cca = CCA(F_g=out_channels, F_x=out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.mamba = MambaBlock(out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 使用CCA进行特征融合
        x2 = self.cca(g=x1, x=x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.mamba(x)
        return x

class CCA(nn.Module):
    """优化后的通道上下文注意力模块"""
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(F_x, F_x, 1),
            nn.BatchNorm2d(F_x)  # 添加批归一化
        )
        self.mlp_g = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(F_g, F_x, 1),
            nn.BatchNorm2d(F_x)  # 添加批归一化
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        channel_att_x = self.mlp_x(x)
        channel_att_g = self.mlp_g(g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0
        scale = torch.sigmoid(channel_att_sum)
        x = x * scale
        return self.relu(x)

class MambaUNet(nn.Module):
    """增强版MambaUNet"""
    def __init__(self, in_channels=3, out_channels=5):
        super().__init__()
        
        # 编码器
        self.enc1 = DownBlock(in_channels, 64)
        self.enc2 = DownBlock(64, 128)
        self.enc3 = DownBlock(128, 256)
        self.enc4 = DownBlock(256, 512)
        
        # 通道变压器
        self.channel_transformer = MambaChannelTransformer(
            channels=[64, 128, 256, 512]
        )
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            MambaBlock(1024)
        )
        
        # 解码器
        self.dec4 = UpBlock(1024, 512)
        self.dec3 = UpBlock(512, 256)
        self.dec2 = UpBlock(256, 128)
        self.dec1 = UpBlock(128, 64)
        
        # 输出层
        self.final = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        # 编码器路径
        x1, p1 = self.enc1(x)
        x2, p2 = self.enc2(p1)
        x3, p3 = self.enc3(p2)
        x4, p4 = self.enc4(p3)
        
        # 通道变压器处理跳跃连接
        x1, x2, x3, x4 = self.channel_transformer([x1, x2, x3, x4])
        
        # 瓶颈
        b = self.bottleneck(p4)
        
        # 解码器路径
        d4 = self.dec4(b, x4)
        d3 = self.dec3(d4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        
        return self.final(d1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 