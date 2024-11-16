import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict


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
        # print(len(conv_outs))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w
        # print(feats.shape)
        ### fuse
        U=sum(conv_outs) #bs,c,h,w
        # print(U.shape)
        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        # print(S.shape)
        Z=self.fc(S) #bs,d
        # print(Z.shape)
        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            # print(weight.shape)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        # print(len(weights))
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        # print(attention_weughts.shape)
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1
        # print(attention_weughts.shape)
        ### fuse
        V=(attention_weughts*feats).sum(0)
        # print(V.shape)
        return V

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    se = SKNet(channel=512,reduction=8)
    output=se(input)
    # print(output.shape)