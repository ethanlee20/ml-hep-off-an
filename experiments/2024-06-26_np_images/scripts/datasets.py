
import math
from random import uniform
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from helpers import list_dc9


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


def check_arguments(level, train, events_per_dist):

    """
    Make sure initialization arguments are reasonable.
    """

    assert level in {"gen", "det"}
    assert type(train) == bool
    assert type(events_per_dist) == int


def make_data_filepath(train:bool):

    """
    Create the filepath of the consolidated datafile.
    """

    data_filename = "df_train.pkl" if train else "df_test.pkl"
    data_dirpath = Path("../datafiles/consolidated")
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


def make_sampled_df(df, sampling_ratio):

    """
    Make a dataframe bootstrapped from dataframe df.
    
    The same number of events are sampled from each delta C9 value.
    A sampling ratio of one will sample the average number
    of events per delta C9 value from each delta C9 value.
    """

    assert sampling_ratio > 0

    df_by_dc9 = df.groupby("dc9")

    num_dc9_values = len(list_dc9())

    num_events = len(df)

    avg_num_events_per_dc9 = num_events / num_dc9_values

    num_samples_per_dc9 = int(sampling_ratio * avg_num_events_per_dc9)
            
    df_sample = df_by_dc9.sample(n=num_samples_per_dc9, replace=True)

    return df_sample


def make_distributions(df, events_per_dist:int, sampling_ratio=None):
    """
    Create a list of distribution dataframes.
    
    Each distributions is homogenous in delta C9.

    Distributions are bootstrapped from the original data
    if sampling_ratio is given.

    Distributions are taken in sequence from the original data
    if sampling_ratio is not given.

    Distributions with greater than half the specified
    number of events per distribution are kept.
    """
    
    if sampling_ratio:
        df = make_sampled_df(df, sampling_ratio)
 
    df_by_dc9 = df.groupby("dc9")

    dists = []
    
    for _, df_group in df_by_dc9:
        
        dist_start_indices = range(0, len(df_group), events_per_dist)

        for i in dist_start_indices:

            df_dist = df_group.iloc[i:i+events_per_dist]

            if len(df_dist) < events_per_dist/2:
                continue

            dists.append(df_dist)
    
    return dists


def make_averages_df(dists:list[pd.DataFrame]):
    
    """
    Make a dataframe of the averages of the
    distributions given by dists.
    """

    avg_dfs = [df_dist.mean().to_frame().T for df_dist in dists]
    
    df_avgs = pd.concat(avg_dfs)
    
    return df_avgs


def make_check_df(dists:list[pd.DataFrame]):

    """
    Make a dataframe to sanity check the method.

    Dataframe will be fake data.
    Dataframe will have incremental values.
    """

    check_dfs = [
        pd.DataFrame(
            {
                "q_squared":[float(0)], 
                "costheta_mu":[float(0)], 
                "costheta_K":[float(0)], 
                "chi":[float(0)], 
                "dc9":[float(0)]
            }
        ) 
        for i in range(1999)
    ]

    df_check = pd.concat(check_dfs)
    
    return df_check


class AvgsDataset(Dataset):
    
    def __init__(self, level:str, train:bool, events_per_dist:int, sampling_ratio=None, std_scale=True):
    
        check_arguments(level, train, events_per_dist)

        data_filepath = make_data_filepath(train)

        df = load_dataframe(data_filepath, level)

        dists = make_distributions(df, events_per_dist, sampling_ratio)

        df_avgs = make_averages_df(dists)
        
        if std_scale:
            df_avgs = scale_std(
                df_avgs, 
                ["q_squared", "costheta_mu", "costheta_K", "chi"]
            )

        self.data = df_avgs

        return
    
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        row = self.data.iloc[idx]
        
        features = row[["q_squared", "costheta_mu", "costheta_K", "chi"]]
        features_np = features.to_numpy()
        features_torch = torch.from_numpy(features_np)
        
        label = row["dc9"]

        return features_torch, label


def make_2d_hist_array(df):

    """
    Make an array composed of two dimensional histograms
    of combinations of variables.

    The final array is three dimensional.
    """

    n_bins = 5
    bins = {
        "q_squared": np.array([0, 1, 6, 12, 16, 20]),
        "chi": np.linspace(start=0, stop=2*np.pi, num=n_bins+1),
        "costheta_mu": np.linspace(start=-1, stop=1, num=n_bins+1),
        "costheta_K": np.linspace(start=-1, stop=1, num=n_bins+1),
    }

    combinations = [
        ["q_squared", "costheta_mu"],
        ["q_squared", "costheta_K"],
        ["q_squared", "chi"],
        ["costheta_K", "chi"],
        ["costheta_mu", "chi"],
        ["costheta_K", "costheta_mu"]
    ]

    hists = [
        np.expand_dims(
            np.histogram2d(df[c[0]], df[c[1]], bins=(bins[c[0]], bins[c[1]]))[0],
            axis=0
        )
        for c in combinations
    ]
    hist_arr = np.concatenate(hists, axis=0)
    # breakpoint()
    
    return hist_arr


class Hist2dDataset(Dataset):
    
    def __init__(self, level:str, train:bool, events_per_dist:int, sampling_ratio=None, std_scale=True):
    
        check_arguments(level, train, events_per_dist)

        data_filepath = make_data_filepath(train)

        df = load_dataframe(data_filepath, level)

        dists = make_distributions(df, events_per_dist, sampling_ratio)

        hists = [make_2d_hist_array(d) for d in dists] 

        self.dists = dists
        self.data = hists


        return
    
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        features = self.data[idx]
        features_torch = torch.from_numpy(features)
        
        label = self.dists[idx]["dc9"].iloc[0].item()

        return features_torch, label
    

def test():
    # d = AvgsDataset("gen", train=True, events_per_dist=3, num_dists_per_dc9=3)
    # l = d.__len__()
    # i = d.__getitem__(3)
    d = AvgsDataset("gen", train=True, events_per_dist=24_000, sampling_ratio=None)
    breakpoint()
    d = AvgsDataset("gen", train=False, events_per_dist=24_000, sampling_ratio=2.0)
    l = d.__len__()
    i = d.__getitem__(3)
    # breakpoint()


if __name__ == "__main__":
    test()