import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from model.fusionModel.unet_parts import DoubleConv

class MedSAM(nn.Module):
    def __init__(self, in_channels=3, output_channels=5, image_size=384):
        super(MedSAM, self).__init__()
        
        # 多尺度特征提取
        self.multi_scale_encoder = MultiScaleViT(
            img_size=image_size,
            patch_sizes=[16, 8, 4],
            in_channels=in_channels,
            embed_dim=768
        )
        
        # 医学图像特定的注意力模块
        self.medical_attention = nn.Sequential(
            nn.Conv2d(768, 768, 3, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Conv2d(768, 768, 1),
            nn.Sigmoid()
        )
        
        # 修改解码器以确保输出尺寸为384x384
        self.decoder = nn.Sequential(
            # 从24x24升采样到48x48
            nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            # 从48x48升采样到96x96
            nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            
            # 从96x96升采样到192x192
            nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            
            # 从192x192升采样到384x384
            nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            
            # 最终输出层
            nn.Conv2d(48, output_channels, kernel_size=1)
        )

    def forward(self, x):
        # 特征提取
        features = self.multi_scale_encoder(x)
        
        # 注意力
        attention = self.medical_attention(features)
        enhanced = features * attention
        
        # 解码
        output = self.decoder(enhanced)
        
        # 确保输出尺寸正确
        if output.shape[-2:] != x.shape[-2:]:
            output = F.interpolate(output, size=x.shape[-2:], mode='bilinear', align_corners=False)
            
        return output

class MultiScaleViT(nn.Module):
    """多尺度Vision Transformer"""
    def __init__(self, img_size, patch_sizes, in_channels, embed_dim):
        super().__init__()
        self.patch_encoders = nn.ModuleList()
        
        # 为不同尺度的patch使用不同的embed_dim，确保输出特征图大小一致
        for i, ps in enumerate(patch_sizes):
            out_dim = embed_dim // (2**i)  # 逐级减小特征维度
            self.patch_encoders.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_dim, kernel_size=ps, stride=ps),
                    nn.LayerNorm([out_dim, img_size//ps, img_size//ps]),
                    nn.GELU()
                )
            )
        
        # 特征融合
        total_dims = sum([embed_dim//(2**i) for i in range(len(patch_sizes))])
        self.fusion = nn.Sequential(
            nn.Conv2d(total_dims, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 获取多尺度特征
        features = []
        for encoder in self.patch_encoders:
            feat = encoder(x)
            features.append(feat)
            
        # 确保所有特征图大小一致
        target_size = features[0].shape[-2:]
        aligned_features = []
        for feat in features:
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        # 拼接并融合特征
        concat_features = torch.cat(aligned_features, dim=1)
        return self.fusion(concat_features)

class MedicalAttentionModule(nn.Module):
    """创新点2: 医学图像特定的注意力机制"""
    def __init__(self, dim, reduction):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 医学特定的边缘注意力
        self.edge_attention = nn.Sequential(
            nn.Conv2d(dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        
        # 空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial = torch.cat([max_pool, avg_pool], dim=1)
        spatial = self.spatial_attention(spatial)
        x = x * spatial
        
        # 边缘注意力
        edge = self.edge_attention(x)
        x = x * edge
        return x

class FeatureFusionModule(nn.Module):
    """创新点3: 特征融合模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.adaptive_fusion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        identity = x
        x = self.conv1x1(x)
        fusion_weights = self.adaptive_fusion(identity)
        return x * fusion_weights

class BoundaryEnhancementModule(nn.Module):
    """创新点4: 边界增强模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.edge_detect = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        edge_features = self.edge_detect(x)
        enhanced = self.edge_enhance(torch.cat([x, edge_features], dim=1))
        return x * enhanced

class AdaptiveMaskDecoder(nn.Module):
    """创新点5: 自适应解码器"""
    def __init__(self, transformer_dim, output_channels):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(transformer_dim // (2**i))
            for i in range(4)
        ])
        
        self.final_conv = nn.Conv2d(transformer_dim//16, output_channels, 1)
        
        # 自适应权重生成
        self.adaptive_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(transformer_dim, 4, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # 生成自适应权重
        weights = self.adaptive_weights(x)
        
        features = []
        for i, block in enumerate(self.decoder_blocks):
            x = block(x)
            features.append(x * weights[:, i:i+1])
        
        # 融合多尺度特征
        x = sum(features)
        return self.final_conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(dim*2, dim, 2, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class ImageEncoderViT(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_channels=3, 
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads
            ) for _ in range(depth)
        ])
        
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return x

class PromptEncoder(nn.Module):
    def __init__(self, embed_dim=256, image_embedding_size=(24, 24), 
                 input_image_size=(384, 384)):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        
        self.pe_layer = PositionalEncoding2D(embed_dim)
        
        self.dense_embedder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )

    def get_dense_pe(self):
        return self.pe_layer(self.image_embedding_size)

    def forward(self, points=None, boxes=None, masks=None):
        bs = 1 if points is None else points.shape[0]
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self.pe_layer.pe.device)
        dense_embeddings = self.dense_embedder(self.get_dense_pe().expand(bs, -1, -1, -1))
        return sparse_embeddings, dense_embeddings

class MaskDecoder(nn.Module):
    def __init__(self, transformer_dim=768, num_multimask_outputs=1, 
                 output_channels=5):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs
        
        self.transformer = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_dim, transformer_dim)
        )
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim//4, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(transformer_dim//4, transformer_dim//8, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(transformer_dim//8, transformer_dim//16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(transformer_dim//16, output_channels, kernel_size=1)
        )

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings):
        # 简化版decoder，直接上采样到原始分辨率
        masks = self.output_upscaling(image_embeddings)
        return masks

class PatchEmbed(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self._attention_block(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def _attention_block(self, x):
        x = x.reshape(x.shape[0], -1, x.shape[-1]).permute(1, 0, 2)
        x = self.attn(x, x, x)[0]
        x = x.permute(1, 0, 2).reshape(x.shape[1], *x.shape[2:])
        return x

class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        pe = torch.zeros((embed_dim, 384, 384))
        y_position = torch.ones(384).unsqueeze(0).expand(384, -1).float()
        x_position = y_position.transpose(0, 1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        pe[0::2, :, :] = torch.sin(x_position * div_term.unsqueeze(-1).unsqueeze(-1))
        pe[1::2, :, :] = torch.cos(y_position * div_term.unsqueeze(-1).unsqueeze(-1))
        
        self.register_buffer('pe', pe)

    def forward(self, size: Tuple[int, int]):
        return self.pe[:, :size[0], :size[1]].unsqueeze(0) 