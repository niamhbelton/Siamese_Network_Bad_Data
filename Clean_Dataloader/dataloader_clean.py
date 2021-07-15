import numpy as np
import os
import torch
import torch.utils.data as data
import pandas as pd


TRAIN_ANOM = [3, 275, 544, 582, 864]
VALID_ANOM = [1159 - 1130, 1230 -1130]

class Dataset(data.Dataset):
    def __init__(self, root_dir, task, plane, train = False ):
        super().__init__()
        self.task = task
        self.plane = plane
        self.root_dir = root_dir
        self.train=train
        if self.train == True:
            self.folder_path = self.root_dir + 'train/{0}/'.format(plane)
            self.records = pd.read_csv(
                self.root_dir + 'train-{0}.csv'.format(task), header=None, names=['id', 'label'])
            self.records = self.records[~self.records.index.isin(TRAIN_ANOM)]
            
        else:
            self.folder_path = self.root_dir + 'valid/{0}/'.format(plane)

            self.records = pd.read_csv(
                self.root_dir + 'valid-{0}.csv'.format(task), header=None, names=['id', 'label'])
            self.records = self.records[~self.records.index.isin(VALID_ANOM)]


        self.records['id'] = self.records['id'].map(
            lambda i: '0' * (4 - len(str(i))) + str(i))
        self.paths = [self.folder_path + filename +
                      '.npy' for filename in self.records['id'].tolist()]
        self.labels = self.records['label'].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])
        label = self.labels[index]
        label = torch.FloatTensor([label])

        array = np.stack((array,)*3, axis=1)
        array = torch.FloatTensor(array)

        return array, label