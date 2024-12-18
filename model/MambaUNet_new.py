# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .MambaUNet_new_sys import VSSM

logger = logging.getLogger(__name__)

class MambaUnet(nn.Module):
    # def __init__(self, img_size=224, num_classes=21843, zero_head=False, vis=False):
    def __init__(self, 
                 patch_size=4, 
                 in_chans=3, 
                 num_classes=5, 
                 depths=[2, 2, 9, 2], 
                 dims=[96, 192, 384, 768], 
                 d_state=16, 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, 
                 patch_norm=True,
                 use_checkpoint=False, 
                 final_upsample="expand_first",
                 zero_head=False, 
                 vis=False):
        super(MambaUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head

        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.d_state = d_state
        self.drop_rate = drop_rate
        self.mamba_unet =  VSSM(                 
            patch_size=patch_size, 
            in_chans=in_chans, 
            num_classes=num_classes, 
            depths=depths, 
            dims=dims, 
            d_state=d_state, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, 
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint, 
            final_upsample=final_upsample,)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.mamba_unet(x)
        return logits

    def load_from(self, pretain_ckpt):
        pretrained_path = pretain_ckpt
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.mamba_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.mamba_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.mamba_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


if __name__ == "__main__":
    x = torch.randn(1, 3, 384, 384).to("cuda")
    net = MambaUnet(                 
        in_chans=3, 
        num_classes=5,).to("cuda")
    y = net(x)
    print("y.shaoe:", y.shape)