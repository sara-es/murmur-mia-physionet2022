from torch.utils.data import Dataset
import numpy as np
import torch


class TrainDataset(Dataset):
    def __init__(self, recordings, features, labels):
        self.recordings = recordings
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, index):
        recording = self.recordings[index].reshape(1,-1)
        label = self.labels[index]
        label = torch.as_tensor(label).float() 
        features = torch.from_numpy(self.features[index]).float()
        return recording, features, label


class TestDataset(Dataset):
    def __init__(self, recordings, features):
        self.recordings = recordings
        self.features = features

    def __len__(self):
        return len(self.recordings)

    def __getitem__(self, index):
        recording = self.recordings[index].reshape(1,-1)
        features = torch.from_numpy(self.features[index]).float()
        return recording, features