import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry

class ScribblePrompt(nn.Module):
    def __init__(self, in_channels=3, output_channels=5, sam_checkpoint=None):
        super().__init__()
        # 初始化SAM作为backbone
        self.sam = sam_model_registry['vit_h'](checkpoint=sam_checkpoint)
        
        # 图像编码器（使用SAM的图像编码器）
        self.image_encoder = self.sam.image_encoder
        
        # 涂鸦提示编码器
        self.scribble_encoder = ScribbleEncoder(
            in_channels=in_channels,
            hidden_dim=256
        )
        
        # 特征融合模块
        self.fusion_module = FeatureFusionModule(
            dim=256,
            num_heads=8
        )
        
        # 掩码解码器
        self.mask_decoder = MaskDecoder(
            transformer_dim=256,
            num_multimask_outputs=1,
            num_classes=output_channels
        )

    def forward(self, image, scribbles):
        # 1. 图像特征提取
        image_embeddings = self.image_encoder(image)
        
        # 2. 涂鸦特征编码
        scribble_embeddings = self.scribble_encoder(scribbles)
        
        # 3. 特征融合
        fused_features = self.fusion_module(image_embeddings, scribble_embeddings)
        
        # 4. 生成分割掩码
        masks = self.mask_decoder(fused_features)
        
        return masks

class ScribbleEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim//2, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(hidden_dim//2)
        self.norm2 = nn.BatchNorm2d(hidden_dim)
        
        # 位置编码
        self.pos_embedding = PositionalEncoding2D(hidden_dim)
        
    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.pos_embedding(x)
        return x

class FeatureFusionModule(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attention = MultiheadCrossAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)
        
    def forward(self, image_feat, scribble_feat):
        # 调整特征维度
        b, c, h, w = image_feat.shape
        image_feat = image_feat.flatten(2).transpose(1, 2)  # B, HW, C
        scribble_feat = scribble_feat.flatten(2).transpose(1, 2)  # B, HW, C
        
        # 交叉注意力
        attn_out = self.attention(
            self.norm1(image_feat), 
            self.norm1(scribble_feat)
        )
        
        # FFN
        output = self.ffn(self.norm2(attn_out))
        
        # 恢复空间维度
        output = output.transpose(1, 2).reshape(b, c, h, w)
        return output

class MaskDecoder(nn.Module):
    def __init__(self, transformer_dim, num_multimask_outputs, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        self.transformer = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=transformer_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=2
        )
        
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim//2, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(transformer_dim//2, transformer_dim//4, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(transformer_dim//4, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # 解码器处理
        b, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)  # HW, B, C
        x = self.transformer(x, x)
        x = x.permute(1, 2, 0).reshape(b, c, h, w)  # B, C, H, W
        
        # 上采样到原始分辨率
        masks = self.upscale(x)
        return masks

class MultiheadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x, context):
        b, n, c = x.shape
        h = self.num_heads
        
        q = self.to_q(x).reshape(b, n, h, c // h).permute(0, 2, 1, 3)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = k.reshape(b, -1, h, c // h).permute(0, 2, 1, 3)
        v = v.reshape(b, -1, h, c // h).permute(0, 2, 1, 3)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        return self.net(x)

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        y_embed = torch.arange(h, device=x.device).float()
        x_embed = torch.arange(w, device=x.device).float()
        
        y_embed = y_embed / (h-1) * 2 - 1
        x_embed = x_embed / (w-1) * 2 - 1
        
        xx_channel = x_embed.repeat(h, 1)
        yy_channel = y_embed.repeat(w, 1).transpose(0, 1)
        
        # 添加位置信息
        xx_channel = xx_channel.view(1, 1, h, w).repeat(b, c//2, 1, 1)
        yy_channel = yy_channel.view(1, 1, h, w).repeat(b, c//2, 1, 1)
        
        pe = torch.cat([xx_channel, yy_channel], dim=1)
        return x + pe