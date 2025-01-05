import torch
import torch.nn as nn
import torch.nn.functional as F

class Basic_blocks(nn.Module):
    def __init__(self, in_channel, out_channel, decay=1):
        super(Basic_blocks, self).__init__()
        self.conv=nn.Conv2d(in_channel,out_channel,1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x=self.conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        return conv2+x

class DSE(nn.Module):
    def __init__(self, in_channel, decay=2):
        super(DSE, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // decay, in_channel, 1),
            nn.Sigmoid()
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // decay, in_channel, 1),
            nn.Sigmoid()
        )
        self.gpool = nn.AdaptiveAvgPool2d(1)
        self.gapool=nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        gp = self.gpool(x)
        se = self.layer1(gp)
        x=x*se
        gap=self.gapool(x)
        se2=self.layer2(gap)
        return x * se2

class ImprovedSpaceatt(nn.Module):
    def __init__(self, in_channel, decay=2):
        super(ImprovedSpaceatt, self).__init__()
        # Q分支保持不变
        self.Q = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.BatchNorm2d(in_channel // decay),
            nn.Conv2d(in_channel // decay, 1, 1),
            nn.Sigmoid()
        )

        # 改进的K分支：添加局部-全局特征融合
        self.K_local = nn.Sequential(
            # 保持原有的局部特征提取
            nn.Conv2d(in_channel, in_channel // decay, 3, padding=1),
            nn.BatchNorm2d(in_channel//decay),
            nn.Conv2d(in_channel//decay, in_channel//decay, 3, padding=1),
        )
        
        # 轻量级全局上下文
        self.K_global = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel//decay, 1),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.K_fusion = nn.Sequential(
            nn.Conv2d(in_channel//decay, in_channel//decay, 1),
            DSE(in_channel//decay)  # 保持原有的DSE模块
        )

        # V分支保持不变
        self.V = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 3, padding=1),
            nn.BatchNorm2d(in_channel//decay),
            nn.Conv2d(in_channel//decay, in_channel//decay, 3, padding=1),
            DSE(in_channel//decay)
        )

        # 输出层保持不变
        self.sig = nn.Sequential(
            nn.Conv2d(in_channel // decay, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, low, high):
        # 1. 生成空间注意力图
        Q = self.Q(low)
        
        # 2. 增强的K分支特征提取
        K_l = self.K_local(low)  # 局部特征
        K_g = self.K_global(low)  # 全局特征
        K_g = F.interpolate(K_g, size=K_l.shape[-2:], 
                          mode='bilinear', align_corners=True)
        K = self.K_fusion(K_l + K_g)  # 特征融合
        
        # 3. V分支保持不变
        V = self.V(high)
        
        # 4. 注意力机制和输出
        att = Q * K
        att = att@V
        return self.sig(att)


class CSCA_blocks(nn.Module):
    def __init__(self, in_channel, decay=4):
        super(CSCA_blocks, self).__init__()
        # 移除upsample层，使用1x1卷积保持通道数不变
        self.proj = nn.Conv2d(in_channel, in_channel, 1)
        
        # 调整通道数，保持一致性
        self.conv = Basic_blocks(in_channel * 2, in_channel)
        self.catt = DSE(in_channel, decay)
        self.satt = ImprovedSpaceatt(in_channel, decay)
        
    def forward(self, x):
        # 使用投影后的特征作为第二个输入
        proj = self.proj(x)
        concat = torch.cat([x, proj], dim=1)
        point = self.conv(concat)
        plusatt = x + satt
        
        # 添加残差连接
        return x + plusatt  # 确保输出与输入具有相同的通道数和尺寸