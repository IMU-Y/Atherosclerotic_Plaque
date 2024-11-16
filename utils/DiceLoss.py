import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input: torch.Tensor, target):
        N = target.size(0)
        smooth = 1

        target_flat = target.contiguous().view(N, -1)
        input_flat = input.view(N, -1)


        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):
        #input需要使用softmax吗
        input = torch.softmax(input,dim=1)
        #将图像标签转换为one-hot编码
        class_nums = 5
        target_onehot = nn.functional.one_hot(target,class_nums)
        target_onehot = target_onehot.transpose(3,1)#N C H W
        C = target_onehot.shape[1]
        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target_onehot[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss
