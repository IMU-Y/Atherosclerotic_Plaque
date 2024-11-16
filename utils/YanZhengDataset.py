from torch.utils import data
import os
import numpy as np
from skimage import io
import torch


class YanZhengDataset(data.Dataset):
    def __init__(self, root_path='dataset', train=False, transform=None, roi=False):
        super(YanZhengDataset, self).__init__()
        self.train = train
        self.root_path = root_path
        self.transform = transform
        self.roi = roi
        # if roi:
        #     self.contour_device = torch.device('cuda:2')
        #     self.contour_model = torch.load('oct_contour_detection.pth', map_location='cpu').to(self.contour_device)
        self.images = self.__load_data__(train, roi)


    def __getitem__(self, index: int):
        if self.train:
            img = io.imread(os.path.join(self.root_path, self.images[index]))

            # print('gt max:{}, min:{}'.format(np.max(gt), np.min(gt)))
        else:
            img = self.images[index].astype(np.uint8)
            # img:(H,W,C) gt:(H,W)
            # tensor = torch.from_numpy(img).to(self.contour_device) / 255
            # print('tensor max:{}, min:{}'.format(torch.max(tensor), torch.min(tensor)))
            # exit(100)

        if self.transform:
            img = self.transform(img)
            #gt需要正则化吗
            #gt = self.transform(gt)

            # if gt.ndim == 3:
            #     gt = self.transform(gt)[0]
        else:
            img = torch.from_numpy(img.transpose((2, 0, 1))).float()#这里的tranpose是把图像从H W C变成 C H W的形状,上面的transform里面也有隐含的相同操作


        return img

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
            images = np.load(os.path.join(self.root_path,'dcm.npy'))
            return images

