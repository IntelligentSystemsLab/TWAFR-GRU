import pandas as pd
import numpy as np
import csv
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def import_data(path,device):
    path_list = []
    name_dict = dict()
    data_dict = dict()
    
    
    path_list.extend(glob.glob(path+'/*.xlsx'))
    for n, npath in enumerate(path_list):
        name_dict[n] = npath[(len(path)+1):(npath.rfind('xlsx')-1)]
        temp = pd.read_excel(npath)
        data = temp['busy'].values.astype('float64')
        data = torch.tensor(data).float()
        data = data.to(device)
        data_dict[n] = data

    return name_dict, data_dict

class MyData(Dataset):
    def __init__(self, data, seq_length,time_len,device):
        self.sample_list = dict()
        self.label_list = dict()
        self.device = device
        for n in range(len(data) - seq_length - time_len):
            sample = data[n:n+seq_length]
            label = data[n+seq_length+time_len]
            self.sample_list[n] = sample
            self.label_list[n] = label

    def __len__(self):
        return int(len(self.sample_list))

    def __getitem__(self, item):
        sample = self.sample_list[item]
        sample = torch.reshape(sample, (-1, 1))
        label = self.label_list[item]
        sample = sample.to(self.device)
        label = label.to(self.device)
        return sample, label
