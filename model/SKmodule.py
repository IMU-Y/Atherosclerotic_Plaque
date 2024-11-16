import numpy as np
import torch
from torch import nn
from collections import OrderedDict





class SKmodule(nn.Module):

    def __init__(self, channel, reduction=16, L=32):
        super(SKmodule, self).__init__()
        self.d = max(L,channel//reduction)
        self.convs = nn.ModuleList([])
        self.fc = nn.Linear(channel,self.d)
        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x1, x2):
        bs1, c1, h1, w1 = x1.size()
        bs2, c2, h2, w2 = x2.size()
        conv_outs = []
        ### split
        conv_outs.append(x1)
        conv_outs.append(x2)

        feats = torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs) #bs,c,h,w
        ### reduction channel
        S = U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs1,c1,1,1)) #bs,channel

        attention_weughts = torch.stack(weights,0)#k,bs,channel,1,1

        attention_weughts = self.softmax(attention_weughts)#k,bs,channel,1,1
        ### fuse
        V1 = (attention_weughts*feats)[0]
        V2 = (attention_weughts*feats)[1]
        return V1,V2

if __name__ == '__main__':
    x1=torch.randn(4,40,48,48)
    se = SKmodule(channel=40,reduction=16)
    V1,V2 = se(input)
    print(V1.shape,'+',V2.shape)