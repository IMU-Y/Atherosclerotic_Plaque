import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk



class DiceLoss(nn.Module):
    # n_classes，表示类别的数量
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes


    def _one_hot_encoder(self, input_tensor):
        # 创建一个空列表，用于存储每个类别的独热编码张量
        tensor_list = []
        # 遍历类别索引的范围（从0到n_classes-1）
        for i in range(self.n_classes):
            # 创建一个二进制张量，其中与当前类别索引相等的元素被设置为1.0（True），其他元素被设置为0.0（False）
            temp_prob = input_tensor == i  #* torch.ones_like(input_tensor)
            # 将张量添加到列表时，通过unsqueeze(1)添加一个单一维度
            tensor_list.append(temp_prob.unsqueeze(1))
            # 沿着通道维度（dim=1）拼接张量列表
        output_tensor = torch.cat(tensor_list, dim=1)
        # 将输出张量转换为浮点型
        return output_tensor.float()

    def _dice_loss(self, score, target):
        # 将目标张量转换为浮点型
        target = target.float()
        # 平滑项，避免分母为零
        smooth = 1e-5
        # 计算预测和目标的交集
        intersect = torch.sum(score * target)
        # 计算目标的总和，即目标的平方的总和
        y_sum = torch.sum(target * target)
        # 计算预测的总和，即预测的平方的总和
        z_sum = torch.sum(score * score)
        # 计算Dice Loss
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        # 将Dice Loss转换为1减去Dice系数
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        # 如果设置了softmax标志，对输入进行softmax激活
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
            # 将目标标签转换为独热编码形式
        target = self._one_hot_encoder(target)
        # 如果未提供权重，默认将每个类别的权重都设为1
        if weight is None:
            weight = [1] * self.n_classes
        # 检查输入和目标的形状是否匹配
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        # 用于存储每个类别的Dice Loss（1-Dice系数）
        class_wise_dice = []
        # 总体损失
        loss = 0.0
        # 遍历每个类别
        for i in range(0, self.n_classes):
            # 计算当前类别的Dice Loss
            dice = self._dice_loss(inputs[:, i], target[:, i])
            # 将1-Dice系数添加到列表中
            class_wise_dice.append(1.0 - dice.item())
            # 将当前类别的Dice Loss加权并累加到总体损失中
            loss += dice * weight[i]
        # 返回平均损失
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    # 将预测和真实标签中大于0的值设置为1，将它们转换为二进制掩模
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    # 如果预测和真实标签中都有正值
    if pred.sum() > 0 and gt.sum()>0:
        # 计算 Dice 系数和 95% Hausdorff 距离
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    # 如果只有预测中有正值而真实标签中没有
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    # 如果真实标签中有正值而预测中没有
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    # 将输入图像和标签转换为NumPy数组
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    # 如果输入图像为3D，进行逐层预测
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            # 提取当前层的图像
            slice = image[ind, :, :]
            # 如果图像尺寸不等于指定的patch大小，进行插值操作（缩放）
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            # 将预处理后的图像转换为PyTorch张量并推送到GPU上
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            # 设置模型为评估模式，并进行推断
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                # 如果图像尺寸不等于指定的patch大小，进行反插值操作（放大）
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    # 如果输入图像为2D
    else:
        # 将预处理后的图像转换为PyTorch张量并推送到GPU上
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        # 设置模型为评估模式，并进行推断
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    # 计算每个类别的性能指标
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # 如果提供了测试保存路径，保存预测、原始图像和标签图像
    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    # 返回每个类别的性能指标列表
    return metric_list