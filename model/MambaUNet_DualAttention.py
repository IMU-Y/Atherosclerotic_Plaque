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
        # 可选���添加投影层确保残差连接维度匹配
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
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        # 添加双重注意力
        self.dual_attention = DualAttention(out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            MambaBlock(out_channels)
        )
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 在跳跃连接特征上应用双重注意力
        x2 = self.dual_attention(x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return x * output

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return x * output

class DualAttention(nn.Module):
    """双重注意力模块"""
    def __init__(self, channel):
        super().__init__()
        self.channel_att = ChannelAttention(channel)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class MambaUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=5):
        super().__init__()
        
        # 编码器
        self.enc1 = DownBlock(in_channels, 64)
        self.enc2 = DownBlock(64, 128)
        self.enc3 = DownBlock(128, 256)
        self.enc4 = DownBlock(256, 512)
        
        # 添加双重注意力到跳跃连接
        self.dual_att1 = DualAttention(64)
        self.dual_att2 = DualAttention(128)
        self.dual_att3 = DualAttention(256)
        self.dual_att4 = DualAttention(512)
        
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
        
        self.final = nn.Conv2d(64, out_channels, 1)
        
    def forward(self, x):
        # 编码器路径
        x1, p1 = self.enc1(x)
        x2, p2 = self.enc2(p1)
        x3, p3 = self.enc3(p2)
        x4, p4 = self.enc4(p3)
        
        # 应用双重注意力到跳跃连接
        x1 = self.dual_att1(x1)
        x2 = self.dual_att2(x2)
        x3 = self.dual_att3(x3)
        x4 = self.dual_att4(x4)
        
        # 瓶颈
        b = self.bottleneck(p4)
        
        # 解码器路径
        d4 = self.dec4(b, x4)
        d3 = self.dec3(d4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        
        return self.final(d1)