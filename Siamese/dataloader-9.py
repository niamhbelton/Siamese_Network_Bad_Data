import numpy as np
import os
import torch
import torch.utils.data as data
import pandas as pd
import random

MEAN = 58.81274059207973
STDDEV = 48.56406668573295



class MRDataset(data.Dataset):
    def __init__(self, root_dir, plane, indexes, train=True, transform=None, weights=None):
        super().__init__()
        self.plane = plane
        self.root_dir = root_dir
        self.train = train
        self.indexes = indexes
        self.records = pd.read_csv('metadata.csv')
        if self.train:
            self.folder_path = self.root_dir + 'train/{0}/'.format(plane)
            self.records = self.records.loc[self.records['mrnet_split'] == 0] 
            
        else:
            transform = None
            self.folder_path = self.root_dir + 'valid/{0}/'.format(plane)
            self.records = self.records.loc[self.records['mrnet_split'] == 1] 

        self.records['id'] = self.records['id'].map(
            lambda i: '0' * (4 - len(str(i))) + str(i))
        self.paths = [self.folder_path + filename +
                      '.npy' for filename in self.records['id'].tolist()]
        self.transform = transform
       
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        
        indexes = self.indexes
        array = np.load(self.paths[index])
        if self.transform:
            array = self.transform(array)
        else:
            array = np.stack((array,)*3, axis=1)
            array = torch.FloatTensor(array)

        
        ind = np.random.randint(len(indexes) + 1) -1
        while (ind == index):
            ind = np.random.randint(len(indexes) + 1) -1
        array2 = np.load(self.paths[indexes[ind]])
        if self.transform:
            array2 = self.transform(array2)
        else:
            array2 = np.stack((array2,)*3, axis=1)
            array2 = torch.FloatTensor(array2)
            
        label = torch.FloatTensor([0])
        array = (array - MEAN) / STDDEV
        array2 = (array2 - MEAN) / STDDEV

        return array, array2, label






