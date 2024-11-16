from torch.utils import data
import os
import numpy as np
from skimage import io
import torch


class OctDataset(data.Dataset):
    def __init__(self, root_path='dataset', train=False, transform=None):
        super(OctDataset, self).__init__()
        self.train = train
        self.root_path = root_path
        self.transform = transform
        self.images, self.gts = self.__load_data__(train)

    def __getitem__(self, index: int):

        if self.train:
            img = io.imread(os.path.join(self.root_path, self.images[index]))
            gt = io.imread(os.path.join(self.root_path, self.gts[index]), as_gray=True)
        else:
            img, gt = self.images[index].astype(np.uint8), self.gts[index].astype(np.uint8) * 255


        if self.transform:
            img = self.transform(img)
            gt = self.transform(gt)
            # if gt.ndim == 3:
            #     gt = self.transform(gt)[0]
        else:
            img = torch.from_numpy(img.transpose((2, 0, 1))).float()
            gt = torch.from_numpy(gt).float()

        return img, gt

    def __len__(self) -> int:
        if self.train:
            return len(self.images)
        else:
            return self.images.shape[0]

    def __load_data__(self, train):
        if train:
            images_list, gts_list = [], []
            with open(os.path.join(self.root_path, 'train_pair.lst'), 'r') as file:
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
