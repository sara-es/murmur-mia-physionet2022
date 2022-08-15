from torch.utils.data import Dataset
import torch
import numpy as np


class TrainDataset1ch(Dataset):
    def __init__(self, data, dems, labels, transform=None):
        self.data = data
        self.labels = labels
        self.dems = dems
        self.transform = transform


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        assert 0 <= index and index < len(self)

        record = self.data[index]
        dem = self.dems[index]
        dem = torch.from_numpy(dem).float()

        if self.transform:
            record = self.transform(record)

        return record, dem, self.labels[index]


class TestDataset1ch(Dataset):
    def __init__(self, data, dems, transform=None):
        self.data = data
        self.dems = dems
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        assert 0 <= index and index < len(self)

        record = self.data[index]
        dem = self.dems[index]
        dem = torch.from_numpy(dem).float()

        if self.transform:
            record = self.transform(record)

        return record, dem
