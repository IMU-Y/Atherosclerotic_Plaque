from torch.utils import data
import os
import numpy as np
from skimage import io
import torch


class GLCM_PlaqueDataset(data.Dataset):
    def __init__(self, root_path='dataset', train=False, transform=None, roi=False):
        super(GLCM_PlaqueDataset, self).__init__()
        self.train = train
        self.root_path = root_path
        self.transform = transform
        self.roi = roi
        # if roi:
        #     self.contour_device = torch.device('cuda:2')
        #     self.contour_model = torch.load('oct_contour_detection.pth', map_location='cpu').to(self.contour_device)
        self.images, self.gts = self.__load_data__(train, roi)

    # 训练时：在原始feature基础上根据图片路径执行transform
    def getTransformedFeature(self,img_path):
        #aug_data/images/o/0/f/0/0.png
        #
        paths = img_path.split("/")
        orient = int(paths[3])
        flip = int(paths[5])
        index = paths[6][0:1]#0.png
        #读取图片
        feature = np.load(os.path.join(self.root_path, "aug_data","features", "{}.npy".format(str(index))))
        #转换成 16x415x415
        feature = feature.transpose(2, 0, 1)

        orients = [0, 90, 180, 270]#逆时针
        flips = [0, 1, 2]# 1-lr 2-up
        #http://liao.cpython.org/numpy13/
        if orient >0:
            feature = np.rot90(feature, orient/90,axes=(1,2))
        if flip >0:
            feature = np.flip(feature,3-flip)#flipup-1;fliplr-2
        return feature
    #测试时，直接使用feature
    def getFeature(self,index):
        # 读取图片
        feature = np.load(os.path.join(self.root_path, "features", "test","{}.npy".format(str(index))))
        # 转换成 16x415x415
        feature = feature.transpose(2, 0, 1)
        return feature

    def __getitem__(self, index: int):
        if self.train:
            img = io.imread(os.path.join(self.root_path, self.images[index]))
            feature = self.getTransformedFeature(self.images[index])
            # feature = np.array([[[1,2,3],
            #                      [4,5,6]],
            #                     [[7,8,9],
            #                      [10,11,12]]])
            gt = io.imread(os.path.join(self.root_path, self.gts[index]), as_gray=True)
            gt = gt.astype(np.float)
            # print('gt max:{}, min:{}'.format(np.max(gt), np.min(gt)))
        else:
            img, gt = self.images[index].astype(np.uint8), self.gts[index].astype(np.float)
            feature = self.getFeature(index)
            # img:(H,W,C) gt:(H,W)
            # tensor = torch.from_numpy(img).to(self.contour_device) / 255
            # print('tensor max:{}, min:{}'.format(torch.max(tensor), torch.min(tensor)))
            # exit(100)

        if self.transform:
            img = self.transform(img)
            gt = self.transform(gt)
            feature = torch.from_numpy(feature.copy()).float()
            # if gt.ndim == 3:
            #     gt = self.transform(gt)[0]
        else:
            img = torch.from_numpy(img.transpose((2, 0, 1))).float()#这里的tranpose是把图像从H W C变成 C H W的形状,上面的transform里面也有隐含的相同操作
            gt = torch.from_numpy(gt).float()
            gt.unsqueeze_(dim=0)
            feature = torch.from_numpy(feature.copy()).float()

        return img, gt, feature

    def __len__(self) -> int:
        if self.train:
            return len(self.images)
        else:
            return self.images.shape[0]

    def __load_data__(self, train, roi):
        if roi:
            aug_file_name = 'train_pair_roi.lst'
        else:
            aug_file_name = 'train_pair.lst'
        if train:
            images_list, gts_list = [], []
            with open(os.path.join(self.root_path, aug_file_name), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip()
                    img_path, gt_path = line.split(' ')
                    images_list.append(img_path)
                    gts_list.append(gt_path)
            return images_list, gts_list
        else:
            images = np.load(os.path.join(self.root_path, 'images', 'test', 'dcm.npy'))
            gts = np.load(os.path.join(self.root_path, 'gt', 'test', 'nii.npy'))
            return images, gts

