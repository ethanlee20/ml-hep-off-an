

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


class Per_Part_Dataset(Dataset):

    """
    Dataset of particle events.
    """
    
    def __init__(self, level:str, train:bool, std_scale=True):

        data_filepath = make_data_filepath(train)

        self.data = load_dataframe(data_filepath, level)

        # self.data = self.data.sample(n=3, random_state=42)
        self.data = self.data[(self.data["q_squared"] < 10) & (self.data["q_squared"] > 9.9) & (self.data["costheta_mu"] < 0.5) & (self.data["costheta_mu"] > 0.4)]

        if std_scale:
            feature_names = ["q_squared", "costheta_mu", "costheta_K", "chi"]
            self.data = scale_std(self.data, feature_names)


        self.data = self.data.to_numpy()

    def __len__(self):

        return len(self.data)
        
    def __getitem__(self, idx):

        features = self.data[idx, :4]
        features_torch = torch.from_numpy(features)
        
        label = self.data[idx, 4]

        return features_torch, label
    


def test():
    ds = Per_Part_Dataset("gen", train=True, std_scale=True)
    l = len(ds)
    i = ds.__getitem__(4)
    breakpoint()


if __name__ == "__main__":
    test()