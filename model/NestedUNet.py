from torch import nn
from torch.nn import functional as F
import torch
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class NestedUNet(nn.Module):
    def __init__(self, deepsupervision,in_channel,out_channel):
        super().__init__()

        self.deepsupervision = deepsupervision

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2,2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(in_channel, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv0_1 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0]*4+nb_filter[1], nb_filter[0])
        self.sigmoid = nn.Sigmoid()
        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input) # torch.Size([1, 32, 370, 370])
        x1_0 = self.conv1_0(self.pool(x0_0))
        print("x0_0:",x0_0.shape)
        print("x1_0:",x1_0.shape)
        # input is CHW
        diffY = x0_0.size()[2] - x1_0.size()[2]
        diffX = x0_0.size()[3] - x1_0.size()[3]

        a = F.pad(x1_0, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x0_1 = self.conv0_1(torch.cat([x0_0, a], 1))


        x2_0 = self.conv2_0(self.pool(x1_0))
        # input is CHW
        diffY = x1_0.size()[2] - x2_0.size()[2]
        diffX = x1_0.size()[3] - x2_0.size()[3]

        a = F.pad(x2_0, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        x1_1 = self.conv1_1(torch.cat([x1_0, a], 1))
        print("x1_1:",x1_1.shape)


        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        # input is CHW
        diffY = x2_0.size()[2] - x3_0.size()[2]
        diffX = x2_0.size()[3] - x3_0.size()[3]

        a = F.pad(x3_0, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])

        x2_1 = self.conv2_1(torch.cat([x2_0, a], 1))
        print("x2_1:", x2_1.shape)
        print("x2_1up",self.up(x2_1).shape)
        # input is CHW
        diffY = x1_0.size()[2] - x2_1.size()[2]
        diffX = x1_0.size()[3] - x2_1.size()[3]

        a = F.pad(x2_1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])

        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, a], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        # input is CHW
        diffY = x1_0.size()[2] - x2_2.size()[2]
        diffX = x1_0.size()[3] - x2_2.size()[3]

        a = F.pad(x2_2, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, a], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output