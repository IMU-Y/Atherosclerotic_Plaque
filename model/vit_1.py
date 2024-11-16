import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, embed_size, heads):
        super(Attention, self).__init__()
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[2], keys.shape[2], query.shape[2]

        values = values.reshape(N, -1, self.heads, self.head_dim)
        keys = keys.reshape(N, -1, self.heads, self.head_dim)
        queries = query.reshape(N, -1, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.head_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, -1
        )

        out = self.fc_out(out)
        return out

class ViTEncoderBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(ViTEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.attention = Attention(embed_size, heads=heads)

        self.fc = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
        )

    def forward(self, x):
        attention = self.attention(x, x, x, None)
        x = self.norm1(attention + x)
        forward = self.fc(x)
        x = self.norm2(forward + x)
        return x

class ViTEncoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, patch_dim, patch_num, block_num):
        super(ViTEncoder, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.num_layers = num_layers
        self.block_num = block_num

        self.patch_embedding = nn.Linear(patch_dim, embed_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, patch_num, embed_size))
        self.transformer_blocks = nn.ModuleList(
            [ViTEncoderBlock(embed_size, heads) for _ in range(num_layers)]
        )

    def forward(self, img):
        B, C, H, W = img.shape
        patch_num = H * W // (self.patch_dim ** 2)

        x = img.reshape(B, C, -1)  # Reshape to (B, C, patch_num)
        x = self.patch_embedding(x)

        pos_embedding = self.pos_embedding[:, :patch_num, :]
        x += pos_embedding

        for _ in range(self.block_num):
            for layer in self.transformer_blocks:
                x = layer(x)

        return x

# Example usage:
# embed_size, heads, num_layers, patch_dim, patch_num, block_num depend on your specific task and dataset
encoder = ViTEncoder(embed_size=768, heads=12, num_layers=6, patch_dim=16, patch_num=100, block_num=4)
print(encoder)
