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

from model.ASPP_UNet import ASPP_UNet
from model.Att_UNet import Att_UNet
# from model.BAM_TransUnet_SPP import BAM_TransUnet_SPP
# from model.CBAM_TransUnet_SPP import CBAM_TransUnet_SPP
from model.CBAM_TransUnet_SPP2 import CBAM_TransUnet_SPP2
from model.CE_UNet import CE_Net_OCT
from model.CS_UNet import CS_UNet
from model.CS_UNet2 import CS_UNet2
from model.DA_SPP_Down_UNet import DA_SPP_Down_UNet
# from model.DA_SPP_UNet import DA_SPP_UNet
# from model.DA_SPP_UNet2 import DA_SPP_UNet2
# from model.DA_TransUnet_SPP import DA_Trans_Unet_SPP
# from model.DA_TransUnet_SPP3 import DA_TransUnet_SPP3
from model.DA_Trans_Unet_SPP2 import DA_Trans_Unet_SPP2
from model.DA_UNet import DA_UNet
from model.DA_UNet2 import DA_UNet2
from model.ECA_TransUnet_SPP import ECA_TransUnet_SPP
from model.GAM_SPP_TransUnet import GAM_SPP_TransUnet
from model.GLCM_UNet import GLCM_UNet
from model.MyModel import MyModel
from model.MyPureUNet import MyPureUNet
from model.PSP_DA_fusion import PSP_DA_fusion
from model.PSP_SE_SPP_Leaky_TransUnet import PSP_SE_SPP_Leaky_TransUnet
from model.PSP_SE_fusion import PSP_SE_fusion
from model.PSP_TransUnet_ASPP2 import PSP_TransUnet_ASPP2
from model.PSP_fusion import PSP_fusion
from model.PSP_fusion2 import PSP_fusion2
from model.PlaqueNet import PlaqueNet
from model.PlaqueNet_CBAM_IN import PlaqueNet_CBAM_IN
from model.PureTransUNet import PureTransUNet
# from model.Pos_TransUNet import Pos_TransUNet
# from model.PureTransUNet import PureTransUNet
from model.PureUNet import PureUNet
from model.PureUNet_BAM import PureUNet_BAM
from model.PureUNet_SPP import PureUNet_SPP
from model.R2Unet import R2U_Net
from model.SE_SPP_Leaky_TransUnet import SE_SPP_Leaky_TransUnet
# from model.SE_TransUnet_SPP import SE_TransUnet_SPP
# from model.SE_TransUnet_SPP2 import SE_TransUnet_SPP2
# from model.SE_UNet2 import SE_UNet2
from model.SE_UNet import SE_UNet
from model.SE_UNet_CBAM import SE_UNet_CBAM
from model.SE_fusion import SE_fusion
from model.SE_max_fusion import SE_max_fusion
from model.SK_concat_fusion import SK_concat_fusion
from model.SK_fuison_direct import SK_fusion_direct
from model.SK_fusion import SK_fusion
from model.SS_UNet import SS_UNet
from model.ScribblePrompt import ScribblePrompt
# from model.ScConv_SPP_TransUnet import ScConv_SPP_TransUnet
from model.TransUnet1block import TransUnet1block
from model.TransUnet2block import TransUnet2block
from model.TransUnet3block import TransUnet3block
from model.TransUnet4block import TransUnet4block
from model.TransUnet5block import TransUnet5block
from model.TransUnet6block import TransUnet6block
from model.TransUnet7block import TransUnet7block
from model.TransUnet8block import TransUnet8block
from model.TransUnet_ASPP import TransUnet_ASPP
from model.TransUnet_ASPP2 import TransUnet_ASPP2
from model.TransUnet_ASPP_CBAM_decoder import TransUnet_ASPP_CBAM_decoder
from model.TransUnet_DA_SPP import TransUnet_DA_SPP
from model.TransUnet_SE_SPP import TransUnet_SE_SPP
from model.TransUnet_atten_ASPP import TransUnet_atten_ASPP
# from model.TransUnet import TransUnet
from model.Trans_Unet_SPP import Trans_Unet_SPP
from model.UNet import UNet
from model.UNet2 import UNet2
from model.fusion_model import fusion_model
from model.scSE_SPP_TransUnet import scSE_SPP_TransUnet
from model.PureUNet_SK_L import PureUNet_SK_L
from utils.FocalLoss import FocalLossMutiClass
from utils.PlaqueDataset import PlaqueDataset
from utils.PlaqueDataset_val import PlaqueDataset_val
from model.MedSAM import MedSAM
from model.SAM_VMNet import SAM_VMNet
from model.UNetX import UNetX
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
                                                     'PSP_DA_fusion','PSP_fusion', 'PSP_fusion2','PSP_SE_fusion', 'SK_concat_fusion','SE_max_fusion','SE_fusion', 'fusion_model', 'SK_fusion_direct', 'SK_fusion', 'max_fusion_model', 'MedSAM', 'SAM_VMNet', 'UNetX'])
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
        path_checkpoint = "/root/autodl-tmp/plaque/checkpoint/SAM_VMNet/SAM_VMNet_plaque_epoch_18_lr_0.05.pth"  # 断点路径
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
    # create model
    if args.net == 'MyModel':
        model: MyModel = MyModel(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'PlaqueNet':
        model: PlaqueNet = PlaqueNet(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'UNet':
        model: UNet = UNet(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'UNet2':
        model: UNet = UNet2(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'PureUNet':
        model: PureUNet = PureUNet(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'DA_UNet':
        model: DA_UNet = DA_UNet(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'DA_UNet2':
        model: DA_UNet = DA_UNet2(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'CE_UNet':
        model: CE_Net_OCT = CE_Net_OCT(output_channels=args.class_num)
    elif args.net == 'ASPP_UNet':
        model: ASPP_UNet = ASPP_UNet(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'CS_UNet':
        model: CS_UNet = CS_UNet(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'CS_UNet2':
        model: CS_UNet2 = CS_UNet2(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'GLCM_UNet':
        model: GLCM_UNet = GLCM_UNet(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'SS_UNet':
        model: SS_UNet = SS_UNet(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'Att_UNet':
        model: Att_UNet = Att_UNet(img_ch=3, output_ch=args.class_num)
    elif args.net == 'ScribblePrompt':
        model = ScribblePrompt(
            in_channels=3, 
            output_channels=args.class_num,
            sam_checkpoint='/path/to/sam_vit_h.pth'  # 需要指定SAM预训练权重路径
        )
    # elif args.net == 'SE_UNet2':
    #     model: SE_UNet2 = SE_UNet2(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'SE_UNet':
        model: SE_UNet = SE_UNet(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'PlaqueNet_CBAM_IN':
        model: PlaqueNet_CBAM_IN = PlaqueNet_CBAM_IN(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'PureUNet_BAM':
        model: PureUNet = PureUNet_BAM(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'MyPureUNet':
        model: MyPureUNet = MyPureUNet(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'SE_UNet_CBAM':
        model: SE_UNet_CBAM = SE_UNet_CBAM(in_channels=3, output_channels=args.class_num, bilinear=True)
    elif args.net == 'PureUNet_SPP':
        model: PureUNet_SPP = PureUNet_SPP(in_channels=3, output_channels=args.class_num, bilinear=True)
    # elif args.net == 'TransUnet':
    #     model: TransUnet = TransUnet(all_in_channels=3,img=370, output_channels=args.class_num,bilinear=True)
    # elif args.net == 'Pos_TransUNet':
    #     model: Pos_TransUNet = Pos_TransUNet(img_dim=384, in_channels=3, out_put_channels=args.class_num)
    elif args.net == 'PureTransUNet':
        model: PureTransUNet = PureTransUNet(img_dim=384, in_channels=3, out_put_channels=args.class_num)
    # elif args.net == 'DA_SPP_UNet':
    #     model: DA_SPP_UNet = DA_SPP_UNet(in_channels=3, out_put_channels=args.class_num)
    # elif args.net == 'DA_SPP_UNet2':
    #     model: DA_SPP_UNet2 = DA_SPP_UNet2(in_channels=3, out_put_channels=args.class_num)
    elif args.net == 'DA_SPP_Down_UNet':
        model: DA_SPP_Down_UNet = DA_SPP_Down_UNet(in_channels=3, out_put_channels=args.class_num)
    elif args.net == 'Trans_Unet_SPP':
        model: Trans_Unet_SPP = Trans_Unet_SPP(in_channels=3, out_put_channels=args.class_num)
    # elif args.net == 'DA_Trans_Unet_SPP':
    #     model: DA_Trans_Unet_SPP = DA_Trans_Unet_SPP(in_channels=3, out_put_channels=args.class_num)
    # elif args.net == 'DA_Trans_Unet_SPP2':
    #     model: DA_Trans_Unet_SPP2 = DA_Trans_Unet_SPP2(in_channels=3, out_put_channels=args.class_num)
    # elif args.net == 'DA_TransUnet_SPP3':
    #     model: DA_TransUnet_SPP3 = DA_TransUnet_SPP3(in_channels=3, out_put_channels=args.class_num)
    # elif args.net == 'BAM_TransUnet_SPP':
    #     model: BAM_TransUnet_SPP = BAM_TransUnet_SPP(in_channels=3, out_put_channels=args.class_num)
    # elif args.net == 'CBAM_TransUnet_SPP':
    #     model: CBAM_TransUnet_SPP = CBAM_TransUnet_SPP(in_channels=3, out_put_channels=args.class_num)
    # elif args.net == 'SE_TransUnet_SPP':
    #     model: SE_TransUnet_SPP = SE_TransUnet_SPP(in_channels=3, out_put_channels=args.class_num)
    # elif args.net == 'SE_TransUnet_SPP2':
    #     model: SE_TransUnet_SPP2 = SE_TransUnet_SPP2(in_channels=3, out_put_channels=args.class_num)
    elif args.net == 'ECA_TransUnet_SPP':
        model: ECA_TransUnet_SPP = ECA_TransUnet_SPP(in_channels=3, out_put_channels=args.class_num)
    elif args.net == 'CBAM_TransUnet_SPP2':
        model: CBAM_TransUnet_SPP2 = CBAM_TransUnet_SPP2(in_channels=3, out_put_channels=args.class_num)
    elif args.net == 'GAM_SPP_TransUnet':
        model: GAM_SPP_TransUnet = GAM_SPP_TransUnet(in_channels=3, out_put_channels=args.class_num)
    elif args.net == 'ScConv_SPP_TransUnet':
        model: ScConv_SPP_TransUnet = ScConv_SPP_TransUnet(in_channels=3, out_put_channels=args.class_num)
    elif args.net == 'scSE_SPP_TransUnet':
        model: scSE_SPP_TransUnet = scSE_SPP_TransUnet(in_channels=3, out_put_channels=args.class_num)
    elif args.net == 'SE_SPP_Leaky_TransUnet':
        model: SE_SPP_Leaky_TransUnet = SE_SPP_Leaky_TransUnet(in_channels=3, out_put_channels=args.class_num)

    elif args.net == 'TransUnet_SE_SPP':
        model: TransUnet_SE_SPP = TransUnet_SE_SPP(in_channels=3, out_put_channels=args.class_num)
    elif args.net == 'TransUnet_DA_SPP':
        model: TransUnet_DA_SPP = TransUnet_DA_SPP(in_channels=3, out_put_channels=args.class_num)
    elif args.net == 'TransUnet3block':
        model: TransUnet3block = TransUnet3block(in_channels=3, out_put_channels=args.class_num)
    elif args.net == 'TransUnet2block':
        model: TransUnet2block = TransUnet2block(in_channels=3, out_put_channels=args.class_num)
    elif args.net == 'TransUnet1block':
        model: TransUnet1block = TransUnet1block(in_channels=3, out_put_channels=args.class_num)
    elif args.net == 'TransUnet4block':
        model: TransUnet4block = TransUnet4block(in_channels=3, out_put_channels=args.class_num)
    elif args.net == 'TransUnet5block':
        model: TransUnet5block = TransUnet5block(in_channels=3, out_put_channels=args.class_num)
    elif args.net == 'TransUnet6block':
        model: TransUnet6block = TransUnet6block(in_channels=3, out_put_channels=args.class_num)
    if args.net == 'TransUnet7block':
        model: TransUnet7block = TransUnet7block(in_channels=3, out_put_channels=args.class_num)
    if args.net == 'TransUnet8block':
        model: TransUnet8block = TransUnet8block(in_channels=3, out_put_channels=args.class_num)
    if args.net == 'TransUnet_ASPP':
        model: TransUnet_ASPP = TransUnet_ASPP(in_channels=3, out_put_channels=args.class_num)
    if args.net == 'TransUnet_ASPP_CBAM_decoder':
        model: TransUnet_ASPP_CBAM_decoder = TransUnet_ASPP_CBAM_decoder(in_channels=3, out_put_channels=args.class_num)
    if args.net == 'TransUnet_atten_ASPP':
        model: TransUnet_atten_ASPP = TransUnet_atten_ASPP(in_channels=3, out_put_channels=args.class_num)
    if args.net == 'R2U_Net':
        model: R2U_Net = R2U_Net(in_channels=3, out_put_channels=args.class_num)
    elif args.net == 'CE_Net_OCT':
        model: CE_Net_OCT = CE_Net_OCT(output_channels=args.class_num)
    elif args.net == 'fusion_model':
        model1 = TransUnet_ASPP2(in_channels=3, out_put_channels=args.class_num)
        model2 = SE_SPP_Leaky_TransUnet(in_channels=3, out_put_channels=args.class_num)
        model: fusion_model = fusion_model(model1, model2)
    elif args.net == 'max_fusion_model':#效果很好
        model1 = TransUnet_ASPP2(in_channels=3, out_put_channels=args.class_num)
        model2 = SE_SPP_Leaky_TransUnet(in_channels=3, out_put_channels=args.class_num)
        model: fusion_model = fusion_model(model1, model2)
    elif args.net == 'SK_fusion_direct':#16epoch效果最佳
        model1 = TransUnet_ASPP2(in_channels=3, out_put_channels=args.class_num)
        model2 = SE_SPP_Leaky_TransUnet(in_channels=3, out_put_channels=args.class_num)
        model: SK_fusion_direct = SK_fusion_direct(model1, model2)
    elif args.net == 'SK_fusion':
        model1 = TransUnet_ASPP2(in_channels=3, out_put_channels=args.class_num)
        model2 = SE_SPP_Leaky_TransUnet(in_channels=3, out_put_channels=args.class_num)
        model: SK_fusion = SK_fusion(model1, model2)
    elif args.net == 'SE_fusion':
        model1 = TransUnet_ASPP2(in_channels=3, out_put_channels=args.class_num)
        model2 = SE_SPP_Leaky_TransUnet(in_channels=3, out_put_channels=args.class_num)
        model: SE_fusion = SE_fusion(model1, model2)
    elif args.net == 'SK_concat_fusion':
        model1 = TransUnet_ASPP2(in_channels=3, out_put_channels=args.class_num)
        model2 = SE_SPP_Leaky_TransUnet(in_channels=3, out_put_channels=args.class_num)
        model: SK_concat_fusion = SK_concat_fusion(model1, model2)
    elif args.net == 'SE_max_fusion':#效果不好
        model1 = TransUnet_ASPP2(in_channels=3, out_put_channels=args.class_num)
        model2 = SE_SPP_Leaky_TransUnet(in_channels=3, out_put_channels=args.class_num)
        model: SE_max_fusion = SE_max_fusion(model1, model2)

    elif args.net == 'PSP_fusion':#效果不错
        model1 = PSP_TransUnet_ASPP2(in_channels=3, out_put_channels=args.class_num)
        model2 = PSP_SE_SPP_Leaky_TransUnet(in_channels=3, out_put_channels=args.class_num)
        model: PSP_fusion = PSP_fusion(model1, model2)
    elif args.net == 'PSP_fusion2':
        model1 = PSP_TransUnet_ASPP2(in_channels=3, out_put_channels=args.class_num)
        model2 = PSP_SE_SPP_Leaky_TransUnet(in_channels=3, out_put_channels=args.class_num)
        model: PSP_fusion2 = PSP_fusion2(model1, model2)
    elif args.net == 'PSP_SE_fusion':
        model1 = PSP_TransUnet_ASPP2(in_channels=3, out_put_channels=args.class_num)
        model2 = PSP_SE_SPP_Leaky_TransUnet(in_channels=3, out_put_channels=args.class_num)
        model: PSP_SE_fusion = PSP_SE_fusion(model1, model2)
    elif args.net == 'PSP_DA_fusion':
        model1 = PSP_TransUnet_ASPP2(in_channels=3, out_put_channels=args.class_num)
        model2 = PSP_SE_SPP_Leaky_TransUnet(in_channels=3, out_put_channels=args.class_num)
        model: PSP_DA_fusion = PSP_DA_fusion(model1, model2)

    elif args.net == 'TransUnet_ASPP2':
        model: TransUnet_ASPP2 = TransUnet_ASPP2(in_channels=3, out_put_channels=args.class_num)
        # model2 = SE_SPP_Leaky_TransUnet(in_channels=3, out_put_channels=args.class_num)
        # model: fusion_model = fusion_model(model1, model2)

    elif args.net == 'MedSAM':
        model: MedSAM = MedSAM(in_channels=3, output_channels=args.class_num)
    elif args.net == 'SAM_VMNet':
        model = SAM_VMNet(in_channels=3, output_channels=args.class_num)
    elif args.net == 'UNetX':
        model = UNetX(in_channels=3, out_channels=args.class_num)

    transformation = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # transforms.Normalize(mean=( 32.624835/255, 32.548692/255,0.913718/255), std=(55.954032/255, 55.481115/255, 8.453468/255))
    ])
    # load data 这里我的torchvision版本不能传transformation
    train_set = PlaqueDataset(root_path=args.root_path, train=True, transform=transformation, roi=args.roi)
    # train_set = PlaqueDataset(root_path=args.root_path, train=True, transform=None, roi=args.roi)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    # 验证集
    val_set = PlaqueDataset_val(root_path=args.root_path, train=True, transform=transformation, roi=args.roi)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

    # create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    # optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.2, last_epoch=-1)
    # 使用余弦退火学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # 第一次重启的epoch数
        T_mult=2,  # 每次重启后周期乘数
        eta_min=1e-6  # 最小学习率
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
