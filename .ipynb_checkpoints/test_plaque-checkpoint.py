import torch
# from model.MyModel import MyModel
import argparse
from utils.PlaqueDataset import PlaqueDataset
from utils.PlaqueDataset_val import PlaqueDataset_val
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
import os
import torchvision.transforms as transforms
import time
from utils.model_size import modelsize
import torch.nn.functional as F
import cv2
import numpy as np
import PIL.Image as Image
import sklearn.metrics as metrics
from utils.statistics import confusion_matrix_statistics
import pandas as pd
# from model.PureUNet_BAM import PureUNet_BAM
import warnings
# 在代码块中禁用特定警告
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)

writer = SummaryWriter("./tensorboardX")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, help="Device ids.", required=True)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--model", default='checkpoint/SE_SPP_Leaky_TransUnet_plaque_epoch_20_lr_0.05.pth')
parser.add_argument("--remark", default='')
parser.add_argument("--class_num", default=5, type=int, choices=[2, 5])
parser.add_argument("--root_path", default="dataset")
args = parser.parse_args()

device = torch.device("cuda:{}".format(args.gpu))

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# device = torch.device("cuda:{}".format(args.gpu))
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def color_the_image(img):
    H, W = img.shape[0], img.shape[1]
    color_img = np.zeros((H, W, 3))
    # RGB 1红：正常斑块 2绿：纤维斑块 3蓝：脂质斑块 4黄:钙化
    color_img[img == 1, 0] = 255
    color_img[img == 2, 1] = 255
    color_img[img == 3, 2] = 255

    # 4黄:钙化
    color_img[img == 4, 0] = 255
    color_img[img == 4, 1] = 255
    # color_img[img != 0, 2] = 255

    return color_img.astype(np.uint8)


def save_results(prediction, images, gts, batch_index):
    name = args.model.split('.')[0] + args.model.split('.')[1]

    # 可视化颜色展示
    # 0为背景，1为红色，2为绿色，3为蓝色

    original_images_path = os.path.join('result', name + args.remark, 'original_images')
    predict_images_path = os.path.join('result', name + args.remark, 'predict_images')
    label_images_path = os.path.join('result', name + args.remark, 'label_images')
    if not os.path.exists(original_images_path):
        os.makedirs(original_images_path)
    if not os.path.exists(predict_images_path):
        os.makedirs(predict_images_path)
    if not os.path.exists(label_images_path):
        os.makedirs(label_images_path)

    # print('prediction max:{}, min:{}'.format(torch.max(prediction), torch.min(prediction)))
    # print('images max:{}, min:{}'.format(torch.max(images), torch.min(images)))
    # print('gts max:{}, min:{}'.format(torch.max(gts), torch.min(gts)))
    show_preds = prediction.detach().clone().cpu().numpy().transpose((0, 2, 3, 1))[..., 0]  # (N, H, W)
    show_imgs = images.detach().clone().cpu().numpy().transpose((0, 2, 3, 1)) * 255  # (N, H, W, 3)
    show_gts = gts.detach().clone().cpu().numpy().transpose((0, 2, 3, 1))[..., 0]  # (N, H, W)
    # print('show_pred shape:', show_preds.shape)
    # print('show_imgs shape:', show_imgs.shape)
    # print('show_gts shape:', show_gts.shape)
    # print('show_preds max:{}, min:{}'.format(np.max(show_preds), np.min(show_preds)))
    # print('show_imgs max:{}, min:{}'.format(np.max(show_imgs), np.min(show_imgs)))
    # print('show_gts max:{}, min:{}'.format(np.max(show_gts), np.min(show_gts)))

    # exit(100)
    for i in range(show_preds.shape[0]):
        png_name = str(args.batch_size * batch_index + i + 1)
        color_predict = color_the_image(show_preds[i])
        color_gt = color_the_image(show_gts[i])

        cv2.imwrite(os.path.join(original_images_path, '{}.png'.format(png_name)),
                    cv2.cvtColor(show_imgs[i].astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(label_images_path, '{}.png'.format(png_name)),
                    cv2.cvtColor(color_gt, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(predict_images_path, '{}.png'.format(png_name)),
                    cv2.cvtColor(color_predict, cv2.COLOR_RGB2BGR))
    pass


def show_feature_maps(model):
    '''
    展示上采样层的可视化结果
    :param model:
    :return:
    '''
    for i in range(len(model.features)):
        writer.add_images('test/feature_map_up{}'.format(str(i)), model.features[i][0].unsqueeze(dim=1))


def test(model, test_loader):
    # model.device_ids = [0]  # 多gpu训练的模型须添加这一句
    model.to(device)
    model.eval()

    images_nums = 0
    # 几个基本指标
    TP, FP, TN, FN = 0, 0, 0, 0
    # 预测正确的像素数目， 像素总数
    correct_pixels, total_pixels = 0, 0
    # 预测第i类正确的像素数目
    correct_class = torch.zeros((5,))
    # gt为第i类的正确的像素数目
    total_class = torch.zeros((5,))
    # 预测为第i类的像素数目（不论是否正确）
    prediction_class = torch.zeros((5,))

    sum_confusion_matrix = np.zeros((5,5), np.int64)
    print('测试图像：{}'.format(len(test_loader.dataset)))

    imgs = []  # 记录图片名
    normal = [] # 记录正常斑块
    fibrous = [] # 记录纤维斑块
    lipid = [] # 记录脂质斑块
    calcium = [] # 记录钙化斑块
    unnormal = [] # 记录非正常斑块
    fibrous_lipid = [] #记录纤维/脂质
    for batch_index, (images, gts) in enumerate(test_loader):
        images, gts = images.to(device), gts.to(device)
        # print('gts max:{}, min:{}'.format(torch.max(gts), torch.min(gts)))

        with torch.no_grad():
            outputs = model(images)
            # print('pred max:{}, min:{}'.format(torch.max(outputs), torch.min(outputs)))
            # print(gts.shape)
        # 在通道维度softmax
        prediction = torch.argmax(torch.softmax(outputs, dim=1), dim=1).unsqueeze(dim=1)
        print(prediction.shape)
        # print('pred max:{}, min:{}'.format(torch.max(prediction), torch.min(prediction)))

        # 使用sklearn.metrics计算混淆矩阵
        gts_numpy = gts.detach().clone().cpu().flatten().numpy()
        prediction_numpy = prediction.detach().clone().cpu().flatten().numpy()
        # print(gts_numpy.shape)
        print(prediction_numpy.shape)

        # 一张图片的shape 172225 (415 * 415 = 172225)
        size_prediction_one = (prediction_numpy.shape[0] // prediction.shape[0])
        for i in range(prediction.shape[0]):
            imgs.append(str(args.batch_size * batch_index + i + 1) + ".png")
            # 定义蓝色点，绿色点，红色点，黄色点的像素值的个数
            sum_blue = 0
            sum_green = 0
            sum_red = 0
            sum_yellow = 0
            # 截取一张图片
            prediction_one = prediction_numpy[size_prediction_one * i:size_prediction_one * (i + 1)]
            for num in prediction_one:
                if num == 1:
                    # 正常斑块
                    sum_red += 1
                if num == 2:
                    # 纤维成分
                    sum_green += 1
                if num == 3:
                    # 脂质成分
                    sum_blue += 1
                if num == 4:
                    # 钙化成分
                    sum_yellow += 1
            total = sum_red + sum_green + sum_blue + sum_yellow
            if total == 0:
                normal.append(str(0) + "(0)")
                fibrous.append(str(0) + "(0)")
                lipid.append(str(0) + "(0)")
                calcium.append(str(0) + "(0)")
                unnormal.append(str(0) + "(0)")
                fibrous_lipid.append(0)
                continue
            normal.append(str(sum_red) + "({})".format(round(sum_red/total,2) * 100))
            fibrous.append(str(sum_green) + "({})".format(round(sum_green / total, 2) * 100))
            lipid.append(str(sum_blue) + "({})".format(round(sum_blue / total, 2) * 100))
            calcium.append(str(sum_yellow) + "({})".format(round(sum_yellow / total, 2) * 100))
            unnormal.append(str(sum_yellow+sum_blue+sum_green) + "({})".format(round((sum_yellow+sum_blue+sum_green) / total,2) * 100))
            if sum_blue != 0:
                print("{}.png 纤维斑块/脂质斑块:{}".format(args.batch_size * batch_index + i + 1, round(sum_green / sum_blue,2)))
                fibrous_lipid.append(round(sum_green / sum_blue,2))
            else:
                print("{}.png 未检测到脂质斑块!".format(args.batch_size * batch_index + i + 1, batch_index))
                fibrous_lipid.append('null')

        sum_confusion_matrix += metrics.confusion_matrix(gts_numpy, prediction_numpy, labels=[0, 1, 2, 3, 4])

        # 可视化特征图
        # show_feature_maps(model)
        if args.class_num == 2:
            gts[gts > 0] = 1
        save_results(prediction, images, gts, batch_index)

        # statistic
        images_nums += images.shape[0]
        correct_pixels += (prediction == gts).sum().item()
        total_pixels += prediction.nelement()
        # MPA &  MIoU
        for i in range(args.class_num):
            # 统计各项指标
            correct_class[i] += ((prediction == i) & (gts == i)).sum().item()
            total_class[i] += (gts == i).sum().item()
            prediction_class[i] += (prediction == i).sum().item()
        if (batch_index + 1) % 10 == 0:
            show_output: torch.Tensor = prediction[0].detach().clone().cpu()
            show_output = show_output.unsqueeze(dim=0)
            show_gt: torch.Tensor = gts[0].detach().clone().cpu()
            show_gt = show_gt.unsqueeze(dim=0)
            show_imgs = torch.cat([show_output, show_gt], dim=0)
            writer.add_images('test{}/test_{}.png'.format(args.remark, batch_index + 1), show_imgs,
                              global_step=batch_index + 1)
            pass

        # 统计TP FP TN FN
        pseudo_gts = gts
        pseudo_gts[pseudo_gts > 0] = 1
        pseudo_predictions = prediction
        pseudo_predictions[pseudo_predictions > 0] = 1
        TP += ((pseudo_predictions == 1) & (pseudo_gts == 1)).sum().item()
        FP += ((pseudo_predictions == 1) & (pseudo_gts == 0)).sum().item()
        TN += ((pseudo_predictions == 0) & (pseudo_gts == 0)).sum().item()
        FN += ((pseudo_predictions == 0) & (pseudo_gts == 1)).sum().item()

    # 将结果写入excel
    dict = {'Rank': imgs, 'Normal(%)': normal,'Fibrous(%)':fibrous,'Lipid(%)':lipid,'Calcium(%)':calcium,'Unnormal(%)':unnormal,'Fibrous/Lipid':fibrous_lipid}
    df = pd.DataFrame(dict)
    # 保存 dataframe
    # df.to_csv('news.csv')
    df.to_excel("病变血管定量描述结果.xlsx")

    confusion_matrix_statistics(sum_confusion_matrix)
    pass
    smooth = 1e-3
    print('PA:{}'.format(correct_pixels / total_pixels))
    MPA = 1 / args.class_num * sum(correct_class[i].item() / total_class[i].item() for i in range(args.class_num))
    print("MPA:{}".format(MPA))
    MIoU = 1 / args.class_num * sum(
        correct_class[i].item() / (total_class[i].item() + prediction_class[i].item() - correct_class[i].item()) for i
        in range(args.class_num))
    print("MIoU:{}".format(MIoU))
    FWIoU = 1 / total_pixels * sum(
        (total_class[i] * correct_class[i]).item() / (total_class[i] + prediction_class[i] - correct_class[i]).item()
        for i in range(args.class_num))
    print("FWIoU:{}".format(FWIoU))
    sensitivity = TP / (TP + FN)
    print('sensitivity:{}'.format(sensitivity))
    specificity = TN / (FP + TN)
    print('specificity:{}'.format(specificity))
    precision = TP / (TP + FP)
    recall = sensitivity
    f1_score = 2 * (precision * recall) / (precision + recall)
    print('f1_score:{}'.format(f1_score))

    writer.close()


def main():
    # load model
    # model.load_state_dict(checkpoint['model'])

    # 加载整个模型
    model = torch.load(args.model, map_location=torch.device('cpu'))

    # 加载模型参数
    # model = PureUNet_BAM(in_channels=3,output_channels=4,bilinear=True)
    # model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu'))['net'])
    # print(model)

    # load testset
    transformation = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(32.527315/255, 10.353297/255, 1.765299/255), std=(42.158743/255, 26.074159/255, 12.691772/255))
    ])
    test_dataset = PlaqueDataset(root_path=args.root_path, train=False, transform=transformation)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    test(model, test_loader)
    pass


if __name__ == '__main__':
    main()
