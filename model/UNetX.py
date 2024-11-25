import torch
import torch.nn as nn
import torch.nn.functional as F

class XBlock(nn.Module):
    """X型特征提取块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels*2, out_channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//4, out_channels*2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        
        feat = torch.cat([b1, b2], dim=1)
        att = self.attention(feat)
        feat = feat * att
        
        return torch.split(feat, feat.size(1)//2, dim=1)[0]

class DownX(nn.Module):
    """下采样X块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.x_block = XBlock(in_channels, out_channels)
        
    def forward(self, x):
        x = self.pool(x)
        x = self.x_block(x)
        return x

class UpX(nn.Module):
    """上采样X块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, 2, stride=2)
        self.x_block = XBlock(in_channels, out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 处理大小不匹配
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x2, x1], dim=1)
        x = self.x_block(x)
        return x

class UNetX(nn.Module):
    def __init__(self, in_channels=3, out_channels=5, features=[64, 128, 256, 512], use_deep_supervision=False):
        super().__init__()
        self.use_deep_supervision = use_deep_supervision  # 默认关闭深度监督
        
        # 初始特征提取
        self.first = XBlock(in_channels, features[0])
        
        # 下采样路径
        self.down_path = nn.ModuleList()
        for idx in range(len(features)-1):
            self.down_path.append(DownX(features[idx], features[idx+1]))
        
        # 瓶颈层
        self.bottleneck = XBlock(features[-1], features[-1])
        
        # 上采样路径
        self.up_path = nn.ModuleList()
        for idx in range(len(features)-1, 0, -1):
            self.up_path.append(UpX(features[idx], features[idx-1]))
            
        # 最终输出层
        self.final = nn.Sequential(
            nn.Conv2d(features[0], features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], out_channels, 1)
        )
        
        # 深度监督
        self.deep_supervision = nn.ModuleList([
            nn.Conv2d(feat, out_channels, 1)
            for feat in features
        ])
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        # 保存下采样特征
        x1 = self.first(x)
        encoder_features = [x1]
        
        # 下采样路径
        for down in self.down_path:
            encoder_features.append(down(encoder_features[-1]))
            
        # 瓶颈层
        x = self.bottleneck(encoder_features[-1])
        
        # 上采样路径
        for idx, up in enumerate(self.up_path):
            x = up(x, encoder_features[-(idx+2)])
            
        # 最终输出
        output = self.final(x)
        
        # 如果不使用深度监督，直接返回最终输出
        if not self.use_deep_supervision or not self.training:
            return output
            
        # 使用深度监督时的代码（可选）
        deep_outputs = []
        for idx, feat in enumerate(encoder_features):
            deep_outputs.append(
                F.interpolate(
                    self.deep_supervision[idx](feat),
                    size=output.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            )
        return [output] + deep_outputs 