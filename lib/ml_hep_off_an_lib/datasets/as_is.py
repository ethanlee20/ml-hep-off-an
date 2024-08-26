

from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


datafiles_dirpath = Path(r"C:\Users\tetha\Desktop\ml-hep-off-an\experiments\2024-06-26_np_images\datafiles")
consol_datafiles_dirpath = Path(r"C:\Users\tetha\Desktop\ml-hep-off-an\experiments\2024-06-26_np_images\datafiles\consolidated")


def file_info(filepath):
    """Accecpts name or path."""
    filepath = Path(filepath)
    split_name = filepath.name.split('_')
    dc9, trial = float(split_name[1]), int(split_name[2])
    result = {'dc9': dc9, 'trial': trial}
    return result


def list_dc9():
    filenames = list(Path(datafiles_dirpath).glob("*.pkl"))
    dc9s = [file_info(name)["dc9"] for name in filenames]
    result = sorted(set(dc9s))
    return result


def scale_std(df, columns):
    
    """
    Scale specified columns of a dataframe by
    centering at zero and setting variance to 1.

    Return scaled dataframe.
    """

    df_scaled = df.copy()
   
    for col in columns:

        std = df_scaled[col].std()
        mean = df_scaled[col].mean()
        
        df_scaled[col] = (df_scaled[col] - mean) / std
   
    return df_scaled


def make_data_filepath(train:bool):

    """
    Create the filepath of the consolidated datafile.
    """

    data_filename = "df_train.pkl" if train else "df_test.pkl"
    data_dirpath = consol_datafiles_dirpath
    data_filepath = data_dirpath.joinpath(data_filename)

    return data_filepath


def load_dataframe(filepath, level):
    
    """
    Load a datafile at a particular simulation level.
    Level can be "gen" for generator level
    or "det" for detector level.
    """
    
    assert level in {"gen", "det"}

    df = pd.read_pickle(filepath).loc[level]
    
    return df


def make_sampled_dist_df(df, num_events_per_dist, num_dists_per_dc9):

    """
    Make a dataframe bootstrapped from dataframe df.
    
    The same number of events are sampled from each delta C9 value.
    A sampling ratio of one will sample the average number
    of events per delta C9 value from each delta C9 value.
    """
    
    assert num_events_per_dist > 0
    assert type(num_events_per_dist) == int
    assert num_dists_per_dc9 > 0
    assert type(num_dists_per_dc9) == int

    df_by_dc9 = df.groupby("dc9")
            
    df_sample = df_by_dc9.sample(n=num_events_per_dist*num_dists_per_dc9, replace=True)

    return df_sample


def make_distributions(df, num_events_per_dist:int, num_dists_per_dc9:int):
    """
    Create a list of distribution dataframes.
    
    Each distribution is homogenous in delta C9.

    Distributions are bootstrapped from the original data.
    """
    
    df = make_sampled_dist_df(df, num_events_per_dist, num_dists_per_dc9)
 
    df_by_dc9 = df.groupby("dc9")

    dists = []
    
    for _, df_group in df_by_dc9:
        
        dist_start_indices = range(0, len(df_group), num_events_per_dist)

        for i in dist_start_indices:

            df_dist = df_group.iloc[i:i+num_events_per_dist]

            if len(df_dist) < num_events_per_dist/2:
                continue

            dists.append(df_dist)
    
    return dists


class As_Is_Dataset(Dataset):

    """
    Dataset of bootstrapped sets of events.
    """
    
    def __init__(self, level:str, train:bool, num_events_per_dist:int, num_dists_per_dc9:int, std_scale=True):

        data_filepath = make_data_filepath(train)

        self.df = load_dataframe(data_filepath, level)

        self.data = make_distributions(self.df, num_events_per_dist, num_dists_per_dc9)

        if std_scale:
            feature_names = ["q_squared", "costheta_mu", "costheta_K", "chi"]
            self.data = [scale_std(d, feature_names) for d in self.data]
    
    def __len__(self):

        return len(self.data)
        
    def __getitem__(self, idx):

        dist_df = self.data[idx]

        dist_np = dist_df.to_numpy()

        features = dist_np[:, :4]
        features_torch = torch.from_numpy(features)
        
        label = dist_np[:, 4, None]

        return features_torch, label
    


def test():
    ds = As_Is_Dataset("gen", train=True, num_events_per_dist=10, num_dists_per_dc9=3, std_scale=True)
    l = len(ds)
    i = ds.__getitem__(4)
    breakpoint()


if __name__ == "__main__":
    test()