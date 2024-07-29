
from math import sqrt
from statistics import mean

from pathlib import Path
import numpy as np
import pandas as pd


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
