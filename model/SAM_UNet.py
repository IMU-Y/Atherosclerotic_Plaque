import torch
import torch.nn as nn
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam

class SAM_UNet(nn.Module):
    def __init__(self, in_channels=3, output_channels=2):
        super(SAM_UNet, self).__init__()
        # 加载预训练SAM模型
        self.sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
        
        # 冻结SAM主干网络参数
        for param in self.sam.parameters():
            param.requires_grad = False
            
        # 添加适配层，将SAM特征转换为所需的输出通道
        self.adapter = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_channels, kernel_size=1)
        )
        
    def forward(self, x):
        # 确保输入尺寸为1024x1024
        if x.shape[-2:] != (1024, 1024):
            x = nn.functional.interpolate(x, size=(1024, 1024), mode='bilinear', align_corners=True)
        
        # 获取SAM图像编码器的特征
        with torch.no_grad():
            features = self.sam.image_encoder(x)
        
        # 通过适配层得到分割结果
        out = self.adapter(features)
        
        # 将输出调整回原始尺寸
        if out.shape[-2:] != x.shape[-2:]:
            out = nn.functional.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=True)
            
        return out