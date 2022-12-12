import torch
import numpy as np
import os

from torch.utils.data import Dataset
from .imagefolder_custom import ImageFolder_custom

class TinyImageNet(Dataset):

    def __init__(self, root, train, transform=None, target_transform=None):

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.dataobj, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        dataobj = ImageFolder_custom(os.path.join(self.root, f'tiny-imagenet-200/{self.train}/'), self.train, self.transform, self.target_transform);

        # data = dataobj.data
        target = np.array(dataobj.target)

        return dataobj, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.dataobj[gs_index, :, :, 1] = 0.0
            self.dataobj[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataobj[index]

        # if self.transform is not None:
        #     img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.dataobj)