
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset


def load_data(in_dir_path, level, split):
    
    in_dir_path = Path(in_dir_path)
    file_base_name = f"{level}_{split}" 

    feature_filename = file_base_name + "_feat.npy"
    label_filename = file_base_name + "_label.npy"
    
    feature_filepath = in_dir_path.joinpath(feature_filename)        
    label_filepath = in_dir_path.joinpath(label_filename)
    
    features = np.load(feature_filepath, allow_pickle=True)
    labels = np.load(label_filepath, allow_pickle=True)

    return features, labels


class Gnn_Input_Dataset(Dataset):
    """
    Dataset of bootstrapped labeled sets.
    """
    def __init__(self, in_dir_path, level, split, device):
        
        features, labels = load_data(in_dir_path, level, split)

        features_torch = torch.from_numpy(features)
        labels_torch = torch.from_numpy(labels)

        self.x = features_torch
        self.y = labels_torch

        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx] # unsqueeze? 
        return x, y
    