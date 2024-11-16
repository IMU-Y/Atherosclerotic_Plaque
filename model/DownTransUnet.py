import torch
import torch.nn as nn

from model.my_unet_parts import DoubleConv
from model.unet_parts import Up, OutConv
from model.vit import ViT
from einops import rearrange


class ResidualBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super(ResidualBlockV2,self).__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)
        _, _, h, w = x.size()
        out = out[:, :, :h//2, :w//2]
        x_down=x_down[:, :, :h//2, :w//2]
        x = x + x_down
        x = self.relu(x)

        return x
    


class Encoder(nn.module):
    def __init__(self):
        super().__init__(self, all_in_channels, img_dim, in_channels=512, embedding_dim=1024, head_num=8, mlp_dim=1024,
                 block_num=6, patch_dim=2, classification=True, num_classes=5, bilinear=True, output_channels=5)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.encoder1 = ResidualBlockV2(64, 128, stride=2)
        self.encoder2 = ResidualBlockV2(128, 256, stride=2)
        self.encoder3 = ResidualBlockV2(256, 512, stride=2)
        self.input_vit=img_dim//2//2//2
        self.vit = ViT(img_dim=self.input_vit, in_channels=512, embedding_dim=1024, head_num=8, mlp_dim=1024,
                 block_num=6, patch_dim=2, classification=False, num_classes=5)

        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)

    def forward(self, x):
        # 64*370*370
        x0 = self.conv1(x)
    
        # 128*185*185
        x1 = self.encoder1(x0)
        # 256*92*92
        x2 = self.encoder2(x2)
        # 512*46*46
        x3 = self.encoder3(x2)

        x4 = self.vit(x3)
        # 1024*23*23
        x4 = rearrange(x4, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)
        # 512*23*23
        x4 = self.conv2(x4)
        x4 = self.norm2(x4)
        # 512*23*23
        x5 = self.relu(x4)

        return x0, x1, x2, x3, x5
    
class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels,bilinear=True):
        super().__init__()
        if bilinear:
            self.upsample = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
            self.layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        

    def forward(self, x1, x2=None):
        x = self.upsample(x1)
        if x2 is not None:
            diffY = x2.size()[2] - x.size()[2]
            diffX = x2.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x], dim=1)

        x = self.layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self,  out_put_channels):
        super().__init__()

        self.decoder1 = DecoderBottleneck(512, 256)
        self.decoder2 = DecoderBottleneck(256, 128)
        self.decoder3 = DecoderBottleneck(128, 64)
        self.decoder4 = DecoderBottleneck(64, 64)

        self.out_conv = nn.Conv2d(64, out_put_channels, kernel_size=1)

    def forward(self, x0, x1, x2, x3, x5):
        x = self.decoder1(x5, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x, x0)
        x = self.out_conv(x)

        return x





class DownTransUnet(nn.Module):
    def __init__(self, all_in_channels,img_dim,out_put_channels):
        super().__init__()
        self.encoder = Encoder( all_in_channels,img_dim, in_channels=512, embedding_dim=1024, head_num=8, mlp_dim=1024,
                 block_num=6, patch_dim=2, classification=True, num_classes=5, bilinear=True, output_channels=5)

        self.decoder = Decoder(out_put_channels)

    def forward(self, x):
        x0, x1, x2, x3, x5 = self.encoder(x)
        x = self.decoder(x0, x1, x2, x3, x5)

        return x







