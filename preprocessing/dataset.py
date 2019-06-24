from torch.utils.data import Dataset
import numpy as np
import torch


class UDataSet(Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def __getitem__(self, item):
        return torch.FloatTensor(self.imgs[item][np.newaxis, :, :])

    def __len__(self):
        return len(self.imgs)
