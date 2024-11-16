from model.unet_parts import *
from collections import OrderedDict

class PureUNet_SK_R(nn.Module):
    def __init__(self, in_channels=3, output_channels=2, bilinear=True):
        super(PureUNet_SK_R, self).__init__()
        self.in_channel = in_channels
        self.bilinear = bilinear#ConvTranspose2d的效果不如Upsample

        #四次下采样
        self.inc = DoubleConv(in_channels, 64)
        self.sk0 = SKNet(64)
        self.down1 = Down(64, 128)
        self.sk1 = SKNet(128)
        self.down2 = Down(128, 256)
        self.sk2 = SKNet(256)
        self.down3 = Down(256, 512)
        self.sk3 = SKNet(512)

        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024// factor)

        self.conv = DoubleConv(1024 // factor, 1024 // factor)

        #四次上采样
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.out = OutConv(64,output_channels)

    def forward(self, x):
        x0 = self.sk0(self.inc(x))
        #print(x0.shape) # torch.Size([1, 64, 415, 415])
        x1 = self.sk1(self.down1(x0))
        #print(x1.shape) # torch.Size([1, 128, 207, 207])
        x2 = self.sk2(self.down2(x1))
        #print(x2.shape) # torch.Size([1, 256, 103, 103])
        x3 = self.sk3(self.down3(x2))
        #print(x3.shape) # torch.Size([1, 512, 51, 51])
        x4 = self.down4(x3)
        #print(x4.shape) # torch.Size([1, 512, 25, 25])

        x5 = self.conv(x4)
        #print(x5.shape) # torch.Size([1, 512, 25, 25])
        # self.features = []

        x = self.up1(x5, x3)
        #print(x.shape) # torch.Size([1, 256, 51, 51])
        # self.features.append(x)
        x = self.up2(x, x2)
        #print(x.shape) # torch.Size([1, 128, 103, 103])

        # self.features.append(x)
        x = self.up3(x, x1)
        #print(x.shape) # torch.Size([1, 64, 207, 207])

        # self.features.append(x)
        x = self.up4(x, x0)
        #print(x.shape) # torch.Size([1, 64, 415, 415])

        x = self.out(x)
       #print(x.shape) # torch.Size([1, 4, 415, 415])
        # self.features.append(x)
        # x = nn.Sigmoid()(x)
        return x

class SKNet(nn.Module):

    def __init__(self, channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
        super(SKNet,self).__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w
        ### fuse
        U=sum(conv_outs) #bs,c,h,w
        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d
        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1
        ### fuse
        V=(attention_weughts*feats).sum(0)
        return V

