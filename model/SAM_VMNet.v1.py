import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SAMFeatureExtractor(nn.Module):
    """SAM特征提取器"""
    def __init__(self, in_channels=3, embed_dim=384):
        super().__init__()
        # SAM-like特征提取
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim//4, kernel_size=4, stride=4),
            nn.BatchNorm2d(embed_dim//4),
            nn.GELU(),
            nn.Conv2d(embed_dim//4, embed_dim//2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim//2),
            nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        # 多尺度特征增强
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim//4, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])

    def forward(self, x):
        # 基础特征提取
        x = self.patch_embed(x)
        
        # 多尺度特征融合
        multi_features = [conv(x) for conv in self.multi_scale_conv]
        multi_features = torch.cat(multi_features + [x], dim=1)
        return multi_features

class VesselAttention(nn.Module):
    """血管注意力模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//16, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, 7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_att(x)
        x = x * ca
        # 空间注意力
        sa = self.spatial_att(x)
        x = x * sa
        return x

class SAM_VMNet(nn.Module):
    def __init__(self, in_channels=3, output_channels=5):
        super().__init__()
        
        # SAM特征提取
        self.feature_extractor = SAMFeatureExtractor(in_channels)
        
        # 血管增强模块
        self.vessel_enhancement = nn.Sequential(
            nn.Conv2d(672, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            VesselAttention(384)
        )
        
        # 修改解码器结构，确保通道数匹配
        self.decoder = nn.ModuleList([
            # 384 -> 192
            nn.Sequential(
                nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                VesselAttention(192)
            ),
            # 192 -> 96
            nn.Sequential(
                nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                VesselAttention(96)
            ),
            # 96 -> 48
            nn.Sequential(
                nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                VesselAttention(48)
            )
        ])
        
        # 边缘增强
        self.edge_enhancement = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, groups=48),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 1)
        )
        
        # 最终输出
        self.final_conv = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, output_channels, 1)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 保存输入尺寸
        input_size = x.shape[-2:]
        
        features = self.feature_extractor(x)
        x = self.vessel_enhancement(features)
        
        for decoder in self.decoder:
            x = decoder(x)
            
        identity = x
        x = self.edge_enhancement(x)
        x = x + identity  # 残差连接
        
        x = self.final_conv(x)
        
        # 确保输出尺寸与输入一致
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
            
        return x 