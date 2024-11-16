import torch
import torch.nn as nn
import math
from utils.DiceLoss import MulticlassDiceLoss
from utils.FocalLoss import FocalLossMutiClass
device = torch.device("cuda:0")
# device = torch.device("cpu")
class MixedLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MixedLoss, self).__init__()

    def forward(self, input, target, weights=None):

        #focal_weight = torch.FloatTensor([0.02,0.18,0.45,0.35]).to(device)#三种斑块都有
        # focal_weight = torch.FloatTensor([0.05, 0.20, 0.5, 0.25]).to(device)#没有红色斑块
        #0为背景，1为红色，2为绿色，3为蓝色 分别对应 纤维，钙化，脂质
        #focal_weight = torch.FloatTensor([0.25, 0.1575, 0.33, 0.2625]).to(device)
        #weight_test之后的最终weight
        focal_weight = torch.FloatTensor([0.145,0.1796,0.3762,0.2992]).to(device)
        dice_weights = [0.1,0.2,0.4,0.3]

        alpha = 8
        focal_loss = FocalLossMutiClass(weight=focal_weight)
        # dice_loss = MulticlassDiceLoss()
        # loss = alpha * focal_loss(input, target,focal_weight) - math.log(dice_loss(input, target,dice_weights))
        loss = focal_loss(input, target)
        #观察一下，二者的量级差不多
        # loss = focal_loss(input, target) +dice_loss(input, target, weights)
        return loss