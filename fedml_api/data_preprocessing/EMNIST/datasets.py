import logging

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import EMNIST

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class EMNIST_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.split = 'byclass'

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        emnist_dataobj = EMNIST(root=self.root, split=self.split, train=self.train, transform=self.transform,
                                target_transform=self.target_transform, download=self.download)

        if self.train:
            data = emnist_dataobj.data
            target = np.array(emnist_dataobj.targets)
        else:
            data = emnist_dataobj.data
            target = np.array(emnist_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        # img = Image.fromarray(img.T, mode='L')  # EMNIST需要转置
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)