import argparse
# from tensorboardX import SummaryWriter
import os
import time

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

from utils.FocalLoss import FocalLossMutiClass
from utils.PlaqueDataset import PlaqueDataset
from utils.PlaqueDataset_val import PlaqueDataset_val

from model.MambaUNet import MambaUNet
torch.cuda.set_per_process_memory_fraction(0.99)
# torch.cuda.set_per_process_memory_growth(True)

# OSError: [Errno 24] Too many open files: '/home/jpl-wz/unet_practice/aug_data/images/o/180/f/1/290.png'
# import torch.multiprocessing 
# torch.multiprocessing.set_sharing_strategy('file_system')


np.set_printoptions(threshold=np.inf)

# writer = SummaryWriter("./tensorboardX")

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, help="Device ids.", required=True)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.05)
# parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--class_num", default=5, type=int, choices=[2, 4])
parser.add_argument("--net", required=True, choices=['PureUNet_SK_R','NestedUNet','PureUNet_ECA_L','PureUNet_SK_L_SPP','PureUNet_SK_L','PureUNet_CBAM_L_SPP','PureUNet_CBAM_L','PureUNet_BAM_L','PureUNet_SPP','SE_UNet','SE_UNet2','Att_UNet','SS_UNet','MyModel', 'UNet','UNet2','PlaqueNet','PureUNet','DA_UNet','DA_UNet2','CE_UNet','ASPP_UNet','CS_UNet','CS_UNet2','GLCM_UNet','PlaqueNet_CBAM_IN',
                                                     'TransUnet2block','TransUnet3block','TransUnet_DA_SPP','TransUnet_SE_SPP','scSE_SPP_TransUnet','ScConv_SPP_TransUnet','GAM_SPP_TransUnet','SE_TransUnet_SPP','SE_TransUnet_SPP2','CBAM_TransUnet_SPP','BAM_TransUnet_SPP','PureUNet_BAM','MyPureUNet','SE_UNet_CBAM','TransUnet','Pos_TransUNet','PureTransUNet','DA_SPP_UNet','DA_SPP_UNet2','DA_SPP_Down_UNet',
                                                     'TransUnet_ASPP','TransUnet8block','TransUnet7block','TransUnet6block','TransUnet5block','TransUnet1block','TransUnet4block','Trans_Unet_SPP','DA_Trans_Unet_SPP','DA_Trans_Unet_SPP2','DA_TransUnet_SPP3','ECA_TransUnet_SPP','CBAM_TransUnet_SPP2',
                                                     'SE_SPP_Leaky_TransUnet','TransUnet_ASPP2', 'CE_Net_OCT','R2U_Net','TransUnet_atten_ASPP','TransUnet_ASPP_CBAM_decoder','ScribblePrompt',
                                                     'PSP_DA_fusion','PSP_fusion', 'PSP_fusion2','PSP_SE_fusion', 'SK_concat_fusion','SE_max_fusion','SE_fusion', 'fusion_model', 'SK_fusion_direct', 'SK_fusion', 'max_fusion_model', 'MedSAM', 'SAM_VMNet', 'UNetX', 'MambaUNet'])
parser.add_argument("--remark", default='')
parser.add_argument("--root_path", default='dataset')
parser.add_argument("--roi", default=False, type=bool, help='whether use RoI mask')
parser.add_argument("--resume", default=False, type=bool)
parser.add_argument("--deepsupervision", default=False)

args = parser.parse_args()

# 单gpu
device = torch.device("cuda:{}".format(args.gpu))


# 多gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# def show_images_gts(batch_index, images: torch.Tensor, gts: torch.Tensor):
# if (batch_index + 1) % 10 == 0:
# show_image = images.detach().clone().cpu()[0]
# show_gt = gts.detach().clone().cpu()[0]

# writer.add_image('train/image', show_image)
# writer.add_image('train/label', show_gt)
# pass


def train(model, train_loader, optimizer, scheduler, val_loader):
    print(args)

    # 多gpu
    # if torch.cuda.device_count() > 1:
    #    print('Lets use', torch.cuda.device_count(), 'GPUs!')
    #    model = nn.DataParallel(model)

    model.to(device)
    # 加载保存的模型（训练过程中断的模型）
    start_epoch = 0
    if args.resume:
        path_checkpoint = "/root/autodl-tmp/plaque/checkpoint/MambaUNet/MambaUNet_plaque_epoch_6_lr_0.05.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        scheduler.load_state_dict(checkpoint['lr_schedule'])

    # model = torch.load("/home/jpl-wz/unet_practice/checkpoint/PureUNet_BAM_plaque_epoch_3_lr_0.05.pth")

    # model.train()
    start_time = time.time()
    # Loss = nn.BCEWithLogitsLoss()
    # Loss = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.07440527,2.02950491, 1.36635133, 1., 0.93001003]).to(torch.device("cuda:0")),ignore_index=255)
    # Loss = DiceLoss()

    Loss = FocalLossMutiClass()
    # Loss = MixedLoss()
    # for epoch in range(start_epoch + 1, args.epoch + 1):
    for epoch in range(start_epoch + 1, args.epoch + 1):
        # scheduler.step()

        total_loss = 0
        start_time = time.time()
        batch_time = time.time()
        # batch = 1
        model.train()
        for batch_index, (images, gts) in enumerate(train_loader, 0):

            # print("训练集加载第{}个batch的时间:{}".format(batch_index + 1, time.time() - batch_time))
            # if batch_index % 10 == 0:
            #     print(batch_index)
            images, gts = images.to(device), gts.to(device)
            gts.squeeze_(dim=1)
            optimizer.zero_grad()
            outputs = model(images)
            # cross entroy needs (N,H,W) long
            gts = gts.long().squeeze(dim=1)  # (N, H, W)

            # unet++
            if args.deepsupervision:
                # 深监督
                loss = 0
                for output in outputs:
                    loss = Loss(output, gts)
                    total_loss += loss.detach().clone().cpu().item()

                loss /= len(outputs)
                total_loss += loss
            else:
                loss = Loss(outputs, gts)
                total_loss += loss.detach().clone().cpu().item()

            # loss = Loss(outputs, gts)
            # total_loss += loss.detach().clone().cpu().item()
            loss.backward()
            optimizer.step()

            if batch_index % 50 == 0:
                now = time.time()
                print('EPOCH:', epoch, '| batch_index :', batch_index, '| train_loss :', total_loss,
                      '| train time: %.4f' % (now - start_time))
            # print('time:{},epoch:{},batch{},loss:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),epoch, batch,total_loss))
            # batch = batch+1
            batch_time = time.time()
            pass

        scheduler.step()
        
        model.eval()
        batch_time1 = time.time()
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

        sum_confusion_matrix = np.zeros((5, 5), np.int64)
        for batch_index, (imgs, gts) in enumerate(val_loader, 0):
            # print("验证集加载第{}个batch的时间:{}".format(batch_index + 1, time.time() - batch_time1))
            # if batch_index % 10 == 0:
            #     print(batch_index)

            images, gts = imgs.to(device), gts.to(device)
            with torch.no_grad():
                outputs = model(images)
            # cross entroy needs (N,H,W) long
            # 在通道维度softmax
            prediction = torch.argmax(torch.softmax(outputs, dim=1), dim=1).unsqueeze(dim=1)
            # print('pred max:{}, min:{}'.format(torch.max(prediction), torch.min(prediction)))

            # 使用sklearn.metrics计算混淆矩阵
            gts_numpy = gts.detach().clone().cpu().flatten().numpy()
            prediction_numpy = prediction.detach().clone().cpu().flatten().numpy()

            sum_confusion_matrix += metrics.confusion_matrix(gts_numpy, prediction_numpy, labels=[0, 1, 2, 3, 4])
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

            # 统计TP FP TN FN
            pseudo_gts = gts
            pseudo_gts[pseudo_gts > 0] = 1
            pseudo_predictions = prediction
            pseudo_predictions[pseudo_predictions > 0] = 1
            TP += ((pseudo_predictions == 1) & (pseudo_gts == 1)).sum().item()
            FP += ((pseudo_predictions == 1) & (pseudo_gts == 0)).sum().item()
            TN += ((pseudo_predictions == 0) & (pseudo_gts == 0)).sum().item()
            FN += ((pseudo_predictions == 0) & (pseudo_gts == 1)).sum().item()

        # 打印验证集f1 score
        precision = TP / (TP + FP)
        sensitivity = TP / (TP + FN)
        recall = sensitivity
        f1_score = 2 * (precision * recall) / (precision + recall)

        # 保存模型参数等信息
        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
            'lr_schedule': scheduler.state_dict()
        }
        if not os.path.isdir("./checkpoint/{}".format(args.net)):
            os.mkdir("./checkpoint/{}".format(args.net))
        torch.save(checkpoint,
                   "./checkpoint/{}/{}_plaque_epoch_{}_lr_{}{}.pth".format(args.net, args.net, epoch, args.lr,
                                                                           args.remark))

        

        # writer.add_scalar('train_lr_{}_epoch_{}_start_time_{}/loss'.format(args.lr, args.epoch, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))), total_loss, global_step=epoch)
        print('time:{},epoch:{}, tarin_loss:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), epoch,
                                                 total_loss))
        print('time:{},epoch:{}, f1_score:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), epoch,
                                                 f1_score))
                                                 
        # 记录loss
        if not os.path.exists(os.path.join('loss', args.root_path)):
            os.makedirs(os.path.join('loss', args.root_path))
        loss_file = "loss/{}/{}.txt".format(args.root_path,args.net)
        with open(loss_file,"a+") as f:
            f.write('time:{},epoch:{}, train_loss:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), epoch, total_loss)+'\n')
            f.write('time:{},epoch:{}, f1_score:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), epoch, f1_score)+'\n')
            f.close
        torch.save(model, os.path.join('checkpoint',
                                       '{}_plaque_epoch_{}_lr_{}{}.pth'.format(args.net, epoch, args.lr, args.remark)))

    torch.save(model, os.path.join('products',
                                   '{}_plaque_lr_{}_epoch_{}{}.pth'.format(args.net, args.lr, args.epoch, args.remark)))
    # writer.close()
    pass


def main():
    # if args.net == 'UNet':
    #     model: UNet = UNet(in_channels=3, output_channels=args.class_num, bilinear=True)
    # elif args.net == 'fusion_model':
    #     model1 = TransUnet_ASPP2(in_channels=3, out_put_channels=args.class_num)
    #     model2 = SE_SPP_Leaky_TransUnet(in_channels=3, out_put_channels=args.class_num)
    #     model: fusion_model = fusion_model(model1, model2)
    # elif args.net == 'max_fusion_model':#效果很好
    #     model1 = TransUnet_ASPP2(in_channels=3, out_put_channels=args.class_num)
    #     model2 = SE_SPP_Leaky_TransUnet(in_channels=3, out_put_channels=args.class_num)
    #     model: fusion_model = fusion_model(model1, model2)
    # elif args.net == 'SK_fusion_direct':#16epoch效果最佳
    #     model1 = TransUnet_ASPP2(in_channels=3, out_put_channels=args.class_num)
    #     model2 = SE_SPP_Leaky_TransUnet(in_channels=3, out_put_channels=args.class_num)
    #     model: SK_fusion_direct = SK_fusion_direct(model1, model2)
    # elif args.net == 'PSP_fusion':#效果不错
    #     model1 = PSP_TransUnet_ASPP2(in_channels=3, out_put_channels=args.class_num)
    #     model2 = PSP_SE_SPP_Leaky_TransUnet(in_channels=3, out_put_channels=args.class_num)
    #     model: PSP_fusion = PSP_fusion(model1, model2)

    if args.net == 'MambaUNet':
        model = MambaUNet(in_channels=3, out_channels=args.class_num)

    transformation = transforms.Compose([
        transforms.ToTensor(),  # 使用自定义ToTensor
    ])
    # load data 这里我的torchvision版本不能传transformation
    train_set = PlaqueDataset(root_path=args.root_path, train=True, transform=transformation, roi=args.roi)
    # train_set = PlaqueDataset(root_path=args.root_path, train=True, transform=None, roi=args.roi)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # 验证集
    val_set = PlaqueDataset_val(root_path=args.root_path, train=True, transform=transformation, roi=args.roi)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    # create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,  # 降低初始学习率
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=args.epoch,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )
    
    # 使用混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    train(model, train_loader, optimizer, scheduler, val_loader)


if __name__ == '__main__':
    if not os.path.exists('tensorboardX'):
        os.makedirs('tensorboardX')
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    if not os.path.exists('products'):
        os.makedirs('products')
    main()
