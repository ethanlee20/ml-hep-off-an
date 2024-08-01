
from math import sqrt
from statistics import mean

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def file_info(filepath):
    """Accecpts name or path."""
    filepath = Path(filepath)
    split_name = filepath.name.split('_')
    dc9, trial = float(split_name[1]), int(split_name[2])
    result = {'dc9': dc9, 'trial': trial}
    return result


def list_dc9():
    filenames = list(Path("../datafiles").glob("*.pkl"))
    dc9s = [file_info(name)["dc9"] for name in filenames]
    result = sorted(set(dc9s))
    return result 


def make_image_filename(dc9, trial, level):
    name = f"dc9_{dc9}_{trial}_img_{level}.npy"
    return name


def make_image_filename_range(dc9, trial_range, level):
    if dc9 == "all":
        result = [
            make_image_filename(dc9_i, t, level) 
            for dc9_i in list_dc9() 
            for t in trial_range
        ]
        return result
    result = [make_image_filename(dc9, t, level) for t in trial_range]
    return result


def make_edges_filename(dc9, trial, level):
    name = f"dc9_{dc9}_{trial}_edges_{level}.npy"
    return name


def load_df(dc9, trial):
    filename = f"dc9_{dc9}_{trial}_re.pkl"
    filepath = Path("../datafiles").joinpath(filename)
    result = pd.read_pickle(filepath)
    return result


def load_image_all_trials(dc9, level):
    assert level in {"gen", "det"}
    filepaths=list(Path("../datafiles").glob(f"dc9_{dc9}_*_img_{level}.npy"))
    shape = np.load(filepaths[0], allow_pickle=True).shape
    result = np.zeros(shape)
    for fp in filepaths:
        result += np.load(fp, allow_pickle=True)
    return result


def load_df_all_trials(dc9):
    filepaths=list(Path("../datafiles").glob(f"dc9_{dc9}_*_re.pkl"))
    dfs = [pd.read_pickle(path) for path in filepaths]
    result = pd.concat(dfs)
    return result


def stats(x:list, y:list):
    """
    Calculate statistics of y values for each x value.
    Return [x_ticks], [y_mean], [y_stdev]
    """
    
    assert len(x) == len(y)
    
    data = list(zip(x, y))
    
    x_ticks = sorted(set(x))
    
    y_by_x = [[d[1] for d in data if d[0]==tick] for tick in x_ticks]

    def stdev(s:list):
        m = mean(s)
        sq = map(lambda i: (i-m)**2, s)
        result = sqrt(sum(sq) / (len(s)-1))
        return result
    
    y_stdev = list(map(stdev, y_by_x))

    y_mean = list(map(mean, y_by_x))

    return x_ticks, y_mean, y_stdev


class ImageDataset(Dataset):
    def __init__(self, level, train):
        self.data_dir = Path("../datafiles/shawn_images")

        train_trial_range = range(1, 13)
        test_trial_range = range(13, 16)

        filenames = (
            make_image_filename_range(
                dc9="all", 
                trial_range=train_trial_range, 
                level=level
            )
            if train
            else 
            make_image_filename_range(
                dc9="all", 
                trial_range=test_trial_range, 
                level=level
            )
        )

        def to_path(filename):
            return self.data_dir.joinpath(filename)
        
        self.filepaths = [
            to_path(f) 
            for f in filenames 
            if to_path(f).is_file()
        ]

    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        image_path = self.filepaths[idx]
        image = np.load(image_path, allow_pickle=True)
        image = torch.from_numpy(image)
        label = file_info(image_path)["dc9"]
        return image, label


def train_loop(dataloader, model, loss_fn, optimizer, device):
    
    def train(X, y):
        model.train() 
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device).unsqueeze(1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    
    num_batches = len(dataloader)
    losses = [train(X, y) for X, y in dataloader]
    avg_train_loss = sum(losses) / num_batches

    return avg_train_loss


def test_loop(dataloader, model, loss_fn, device):

    def test(X, y):
        model.eval()
        with torch.no_grad():
            pred = model(X.to(device))
            loss = loss_fn(pred, y.to(device).unsqueeze(1))
            return loss.item()
    
    num_batches = len(dataloader)
    losses = [test(X, y) for X, y in dataloader]
    avg_test_loss = sum(losses) / num_batches
    
    return avg_test_loss
