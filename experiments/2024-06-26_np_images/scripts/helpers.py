
import pickle
from math import sqrt
from statistics import mean

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset





def make_image_filename(dc9, trial, level, ext=".npy"):
    name = f"dc9_{dc9}_{trial}_img_{level}{ext}"
    return name


def make_image_filename_range(dc9, trial_range, level, ext=".npy"):
    if dc9 == "all":
        result = [
            make_image_filename(dc9_i, t, level, ext=ext) 
            for dc9_i in list_dc9() 
            for t in trial_range
        ]
        return result
    result = [make_image_filename(dc9, t, level) for t in trial_range]
    return result


def load_df(dc9, trial):
    filename = f"dc9_{dc9}_{trial}_re.pkl"
    filepath = Path("../datafiles").joinpath(filename)
    assert filepath.is_file(), "File doesn't exist?"
    result = pd.read_pickle(filepath)
    return result


def load_df_trial_range(dc9, trial_range:range):
    dfs = [load_df(dc9, t) for t in trial_range]
    df_all = pd.concat(dfs)
    return df_all


def load_trial_range(trial_range:range, data_dirpath="../datafiles/raw"):
    data_dirpath = Path(data_dirpath)

    filepaths = []
    for t in trial_range:
        filepaths += data_dirpath.glob(f"dc9_*_{t}_re.pkl")

    return filepaths


def load_df_all_trials(dc9):
    filepaths=list(Path("../datafiles").glob(f"dc9_{dc9}_*_re.pkl"))
    dfs = [pd.read_pickle(path) for path in filepaths]
    result = pd.concat(dfs)
    return result


def load_image(dc9, trial, level, dirpath):
    dirpath = Path(dirpath)
    filepath = dirpath.joinpath(make_image_filename(dc9, trial, level))
    image = np.load(filepath, allow_pickle=True)
    return image


def load_image_all_trials(dc9, level):
    assert level in {"gen", "det"}
    filepaths=list(Path("../datafiles").glob(f"dc9_{dc9}_*_img_{level}.npy"))
    shape = np.load(filepaths[0], allow_pickle=True).shape
    result = np.zeros(shape)
    for fp in filepaths:
        result += np.load(fp, allow_pickle=True)
    return result


def select_device():
    device = (
        "cuda" 
        if torch.cuda.is_available()
        else 
        "cpu"
    )
    print("Device: ", device)
    return device


