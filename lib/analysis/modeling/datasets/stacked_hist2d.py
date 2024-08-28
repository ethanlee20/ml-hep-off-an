
from pathlib import Path

import numpy as np

from torch import from_numpy
from torch.utils.data import Dataset


def load_data(in_dir_path, split, level):
    
    in_dir_path = Path(in_dir_path)
    file_base_name = f"{level}_{split}" 

    hist_filename = file_base_name + "_hist.npy"
    label_filename = file_base_name + "_label.npy"
    
    hist_filepath = in_dir_path.joinpath(hist_filename)        
    label_filepath = in_dir_path.joinpath(label_filename)
    
    hists = np.load(hist_filepath, allow_pickle=True)
    labels = np.load(label_filepath, allow_pickle=True)

    return hists, labels


class Stacked_Hist2d_Dataset(Dataset):
    """
    Dataset of stacked 2d histograms.
    """
    def __init__(self, in_dir_path, split, level, device):
        
        hists, labels = load_data(in_dir_path, split, level)

        hists_torch = from_numpy(hists)
        labels_torch = from_numpy(labels)

        self.x = hists_torch
        self.y = labels_torch

        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y
    