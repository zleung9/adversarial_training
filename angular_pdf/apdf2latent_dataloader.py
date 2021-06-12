import numpy as np
import math

import torch
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
from torchvision import transforms

class Apdf2LatDataset(Dataset):
    def __init__(self, data_dict):
        super(Apdf2LatDataset, self).__init__()

        descriptor_list = [[k,d['structure'],d['bvs'],d['qvec'],d['cn'],d['apdf']] 
                            for k, d in data_dict.items()]

        self.mpid = [d[0] for d in descriptor_list]
        self.structure = [d[1] for d in descriptor_list]
        self.bvs = np.array([d[2] for d in descriptor_list])
        self.qvec = np.array([d[3] for d in descriptor_list])
        self.cn = np.array([d[4] for d in descriptor_list])
        self.apdf = np.array([d[5] for d in descriptor_list])[:, np.newaxis]

        self.has_partition = False # place holder for partition

        self.weight_per_cn = 1 / np.fabs(self.cn.mean(axis=0)) # more counts --> less weight
        self.cn_weights = (self.cn * self.weight_per_cn).max(axis=1)

    def __len__(self):
        return len(self.mpid)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.apdf[idx], self.cn[idx]
        return sample
        
    def partition(self, ratio=(0.7, 0.15, 0.15)):

        N_samples = len(self.mpid)

        N_train = int(np.floor(N_samples * ratio[0]))
        N_test = int(np.floor(N_samples * ratio[1]))
        N_val = N_samples - N_train - N_test

        index_shuffle = np.random.permutation(range(N_samples))
        self.index_train = index_shuffle[:N_train]
        self.index_test = index_shuffle[N_train: N_train+N_test]
        self.index_val = index_shuffle[-N_val:]

        self.has_partition = True

    def cut_range(self, r=(1.8, 3.0)):
        R = np.linspace(1.6,8,64)
        r_selection = (R>=r[0])&(R<=r[1])
        self.apdf = tensor(self.apdf[:,:,:,r_selection],dtype=torch.float)

class ToTensor(object):
    def __call__(self, sample):
        return torch.Tensor(sample)


def get_Apdf2Lat_dataloaders(dataset, batch_size, ratio=(0.7, 0.15, 0.15)):
    
    dataset.partition(ratio=ratio)
    if not dataset.has_partition:
        print("Data partition failed.")
        return None, None, None
    else:
        ds_train = Subset(dataset, dataset.index_train)
        ds_test = Subset(dataset, dataset.index_test)
        ds_val = Subset(dataset, dataset.index_val)

    train_sampler = WeightedRandomSampler(dataset.cn_weights[dataset.index_train], replacement=True,
                                          num_samples=math.ceil(len(ds_train)/batch_size)*batch_size)

    train_loader = DataLoader(ds_train, batch_size=batch_size, num_workers=0, pin_memory=False,
                              sampler=train_sampler)
    val_loader = DataLoader(ds_val, batch_size=batch_size, num_workers=0, pin_memory=False)
    test_loader = DataLoader(ds_test, batch_size=batch_size, num_workers=0, pin_memory=False)

    return train_loader, val_loader, test_loader
