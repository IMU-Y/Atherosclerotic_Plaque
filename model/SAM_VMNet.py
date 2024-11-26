import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SAM_VMNet(nn.Module):
    def __init__(self, in_channels=3, output_channels=5):
        super().__init__()
        
        # 1. SAM特征提取
        self.feature_extractor = nn.Sequential(
            # 初始特征提取
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 第一个下采样块
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第二个下采样块
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第三个下采样块
            nn.Conv2d(256, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # 2. 多尺度特征融合
        self.multi_scale = MultiScaleModule(384)
        
        # 3. 修改血管增强模块的输入通道
        self.vessel_enhancement = nn.Sequential(
            nn.Conv2d(384, 384, 1),  # 修改输入通道数为384
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            EnhancedVesselAttention(384)
        )
        
        # 4. 改进解码器，添加跳跃连接
        self.decoder = nn.ModuleList([
            DecoderBlockV2(384, 192),
            DecoderBlockV2(192, 96),
            DecoderBlockV2(96, 48)
        ])
        
        # 5. 改进边缘增强模块
        self.edge_enhancement = EdgeEnhancementModule(48)
        
        # 6. 改进最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(48, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
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
        
        # 修改前向传播逻辑
        features = self.feature_extractor(x)
        ms_features = self.multi_scale(features)  # 添加多尺度特征
        x = self.vessel_enhancement(ms_features)  # 直接使用多尺度特征
        
        # 解码过程
        for decoder in self.decoder:
            x = decoder(x)
            
        # 边缘增强
        identity = x
        x = self.edge_enhancement(x)
        x = x + identity  # 残差连接
        
        x = self.final_conv(x)
        
        # 确保输出尺寸与输入一致
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
            
        return x 

class MultiScaleModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels//4, 3, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(channels//4),
                nn.ReLU(inplace=True)
            ) for dilation in [1, 2, 4, 8]
        ])
        
        # 修改融合层以保持通道数
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1),  # 保持通道数不变
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branches = [branch(x) for branch in self.branches]
        concat_features = torch.cat(branches, dim=1)
        return self.fuse(concat_features)

class EnhancedVesselAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        # 空间注意力
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels//8, 1),
            nn.BatchNorm2d(channels//8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//8, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        sa = self.spatial(x)
        ca = self.channel(x)
        return x * sa * ca

class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        
        self.attention = EnhancedVesselAttention(in_channels//2)
        
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels//2, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        return self.conv2(x)

class EdgeEnhancementModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )
        
        self.refine = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        edge = self.edge_conv(x)
        return self.refine(torch.cat([x, edge], dim=1))