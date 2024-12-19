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

class Spaceatt(nn.Module):
    def __init__(self, in_channel, decay=2):
        super(Spaceatt, self).__init__()
        self.Q = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.BatchNorm2d(in_channel // decay),
            nn.Conv2d(in_channel // decay, 1, 1),
            nn.Sigmoid()
        )
        self.K = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 3, padding=1),
            nn.BatchNorm2d(in_channel//decay),
            nn.Conv2d(in_channel//decay, in_channel//decay, 3, padding=1),
            DSE(in_channel//decay)
        )
        self.V = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 3, padding=1),
            nn.BatchNorm2d(in_channel//decay),
            nn.Conv2d(in_channel//decay, in_channel//decay, 3, padding=1),
            DSE(in_channel//decay)
        )
        self.sig = nn.Sequential(
            nn.Conv2d(in_channel // decay, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, low, high):
        Q = self.Q(low)
        K = self.K(low)
        V = self.V(high)
        att = Q * K
        att = att@V
        return self.sig(att)


class CSCA_blocks(nn.Module):
    def __init__(self, in_channel, out_channel, decay=2):
        super(CSCA_blocks, self).__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channel, out_channel, 2, stride=2)
        self.conv = Basic_blocks(in_channel, out_channel//2)
        self.catt = DSE(out_channel//2, decay)
        self.satt = Spaceatt(out_channel//2, decay)
    def forward(self, high, low):
        up = self.upsample(high)
        concat = torch.cat([up, low], dim=1)
        point = self.conv(concat)
        catt = self.catt(point)
        satt = self.satt(point, catt)
        plusatt = catt*satt
        return torch.cat([plusatt, catt], dim=1)
