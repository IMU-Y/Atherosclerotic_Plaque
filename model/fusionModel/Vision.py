import torch
import torch.nn as nn
from einops import rearrange, repeat

class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()
        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x):
        qkv = self.qkv_layer(x)
        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        attention = torch.softmax(energy, dim=-1)
        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_attention(x)

        return x

class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()
        self.layer_blocks = nn.ModuleList([TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)
        return x

class VisionTransformerEncoder(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()
        self.num_patches = (img_dim // patch_dim) ** 2
        self.patch_dim = patch_dim
        self.embedding_dim = embedding_dim

        self.patch_embedding = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_dim, stride=patch_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embedding_dim))
        self.dropout = nn.Dropout(0.1)

        self.transformer_encoder = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        x = x + self.positional_embedding[:, :x.size(1)]
        x = self.dropout(x)

        x = self.transformer_encoder(x)

        return x

# Example usage:
# img_dim = 46
# in_channels = 512
# embedding_dim = 1024
# head_num = 4
# mlp_dim = 1024
# block_num = 6
# patch_dim = 2
# B=64
#
# vit_encoder = VisionTransformerEncoder(img_dim, in_channels, embedding_dim, head_num, mlp_dim, block_num, patch_dim)
# input_tensor = torch.randn(B, in_channels, img_dim, img_dim)
# output = vit_encoder(input_tensor)
# print(output.shape)
# l=rearrange(output, "b (x y) c -> b x y c", x=46//2, y=46//2)
# print(l.shape)

