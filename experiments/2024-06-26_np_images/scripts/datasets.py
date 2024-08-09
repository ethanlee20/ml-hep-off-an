
import math

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


def check_arguments(level, train, events_per_dist, sampling_ratio):

    """
    Make sure initialization arguments are reasonable.
    """

    assert level in {"gen", "det"}
    assert type(train) == bool
    assert type(events_per_dist) == int
    if not train: 
        assert sampling_ratio == None


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

    num_samples_per_dc9 = sampling_ratio * avg_num_events_per_dc9
            
    df_sample = df_by_dc9.sample(n=num_samples_per_dc9, replace=True)

    return df_sample


def make_distributions(df, train:bool, events_per_dist:int, sampling_ratio=None):
    """
    Create a list of distribution dataframes.
    
    Each distributions is homogenous in delta C9.

    Distributions for the training set are bootstrapped
    from the original training data.

    Distributions for the test set are taken in sequence 
    from the training data.
    
    Sampling ratio must be specified for the training set.

    Distributions with greater than half the specified
    number of events per distribution are kept.
    """
    
    if not train:
        assert not sampling_ratio

    if train:
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


class AvgsDataset(Dataset):
    
    def __init__(self, level:str, train:bool, events_per_dist:int, sampling_ratio=None):
    
        check_arguments(level, train, events_per_dist, sampling_ratio)

        data_filepath = make_data_filepath(train)

        df = load_dataframe(data_filepath, level)

        df_by_dc9 = df.groupby("dc9")

        if not train:

            df_avgs = pd.DataFrame(columns=["q_squared", "costheta_mu", "costheta_K", "chi", "dc9"])

            for _, df_group in df_by_dc9:

                for i in range(0, len(df_group), events_per_dist):

                    df_dist = df_group.iloc[i:i+events_per_dist]

                    if len(df_dist) < events_per_dist/2: 
                        continue

                    df_avgs = pd.concat([
                        None if df_avgs.empty 
                        else df_avgs, 
                        df_dist.mean().to_frame().T
                    ])

            df_scaled_avgs = scale_std(df_avgs, ["q_squared", "costheta_mu", "costheta_K", "chi"])

            self.data = df_scaled_avgs
           
            return
        
        if not sampling_ratio: 
            sampling_ratio = 1.0

        num_dc9_values = len(list_dc9())

        num_events = len(df)
            
        num_dists_per_dc9 = int(sampling_ratio * num_events / (events_per_dist * num_dc9_values))
        
        df_sample = df_by_dc9.sample(n=events_per_dist*num_dists_per_dc9, replace=True)

        df_sample_avgs = pd.DataFrame(columns=["q_squared", "costheta_mu", "costheta_K", "chi", "dc9"])

        for i in range(0, len(df_sample), events_per_dist):

            df_dist = df_sample.iloc[i:i+events_per_dist]
            
            df_sample_avgs = pd.concat([
                None if df_sample_avgs.empty 
                else df_sample_avgs, 
                df_dist.mean().to_frame().T
            ]) 

        df_scaled_sample_avgs = scale_std(
            df_sample_avgs, 
            ["q_squared", "costheta_mu", "costheta_K", "chi"]
        )

        self.data = df_scaled_sample_avgs

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
    

def test():
    # d = AvgsDataset("gen", train=True, events_per_dist=3, num_dists_per_dc9=3)
    # l = d.__len__()
    # i = d.__getitem__(3)
    d = AvgsDataset("gen", train=True, events_per_dist=24_000, sampling_ratio=2.0)
    breakpoint()
    d = AvgsDataset("gen", train=False, events_per_dist=24_000,)
    l = d.__len__()
    i = d.__getitem__(3)
    breakpoint()

# test()
