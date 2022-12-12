import torch
import torchvision
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder,DatasetFolder


class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, train, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader

        self.dataobj = imagefolder_obj;
        
        self.samples = np.array(imagefolder_obj.samples)
        self.target = np.array([int(s[1]) for s in self.samples])

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)
