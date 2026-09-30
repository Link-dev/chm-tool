"""PyTorch dataset over chip stacks."""
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import model_input


class ChipDataset(Dataset):
    """(input, label) pairs for chips `idx` of memory-mapped stacks.

    Inputs are returned as float32 raw values (the model normalises them); labels as [1, H, W] float32
    with -999 = missing. With zero_to_nodata, zero-valued labels are also treated as missing.
    """

    def __init__(self, input_name, annual, seasonal, labels, idx, zero_to_nodata=False):
        self.input_name, self.annual, self.seasonal, self.labels = input_name, annual, seasonal, labels
        self.idx, self.zero_to_nodata = np.asarray(idx), zero_to_nodata

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = model_input(self.input_name, self.annual, self.seasonal, j)
        y = np.array(self.labels[j], dtype=np.float32)
        if self.zero_to_nodata:
            y[y == 0] = -999
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y[None], dtype=torch.float32)
