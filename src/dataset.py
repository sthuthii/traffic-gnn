# src/dataset.py
import torch
from torch.utils.data import Dataset

class TrafficDataset(Dataset):
    def __init__(self, X, Y):
        # X: [S, in_len, N], Y: [S, out_len, N]
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # return shapes: X [in_len, N], Y [out_len, N]
        return self.X[idx], self.Y[idx]
