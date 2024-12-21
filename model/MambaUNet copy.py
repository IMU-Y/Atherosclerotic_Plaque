import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F
from .CSCA import CSCA_blocks

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

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.mamba = MambaBlock(out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.mamba(x)
        return x



class MambaUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=5):
        super().__init__()
        
        # 编码器
        self.enc1 = DownBlock(in_channels, 64)
        self.enc2 = DownBlock(64, 128)
        self.enc3 = DownBlock(128, 256)
        self.enc4 = DownBlock(256, 512)
        
        # 多尺度特征处理模块
        self.x4_dem = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_dem = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x2_dem = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x1_dem = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # 特征差异处理模块
        self.x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # 多层级特征融合模块
        self.x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # 特征融合层
        self.level3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.level1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            CSCA_blocks(1024)
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
        
        # 多尺度特征处理
        x4_dem = self.x4_dem(x4)
        x3_dem = self.x3_dem(x3)
        x2_dem = self.x2_dem(x2)
        x1_dem = self.x1_dem(x1)

        # 计算特征差异
        x4_3 = self.x4_x3(abs(F.interpolate(x4_dem, size=x3.size()[2:], mode='bilinear') - x3_dem))
        x3_2 = self.x3_x2(abs(F.interpolate(x3_dem, size=x2.size()[2:], mode='bilinear') - x2_dem))
        x2_1 = self.x2_x1(abs(F.interpolate(x2_dem, size=x1.size()[2:], mode='bilinear') - x1_dem))

        # 多层级特征融合
        x4_3_2 = self.x4_x3_x2(abs(F.interpolate(x4_3, size=x3_2.size()[2:], mode='bilinear') - x3_2))
        x3_2_1 = self.x3_x2_x1(abs(F.interpolate(x3_2, size=x2_1.size()[2:], mode='bilinear') - x2_1))
        x4_3_2_1 = self.x4_x3_x2_x1(abs(F.interpolate(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear') - x3_2_1))

        # 特征融合
        level3 = self.level3(x4_3)
        level2 = self.level2(x3_2 + x4_3_2)
        level1 = self.level1(x2_1 + x3_2_1 + x4_3_2_1)

        # 瓶颈层
        b = self.bottleneck(p4)
        
        # 解码器路径（与多尺度特征融合）
        d4 = self.dec4(b, x4 + level3)
        d3 = self.dec3(d4, x3 + level2)
        d2 = self.dec2(d3, x2 + level1)
        d1 = self.dec1(d2, x1)
        
        return self.final(d1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 