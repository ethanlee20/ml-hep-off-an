
from pathlib import Path

import pandas as pd


def make_agg_data_filepath(split):
    """
    Create the filepath of the aggregated data file.

    Parameters
    ----------
    split : str
        "train" gives the training data filepath.
        "test" gives the testing data filepath.

    Returns
    -------
    pathlib.Path
    """

    assert split in {"train", "test"}

    data_filename = f"df_{split}.pkl"
    data_dirpath = Path(r"C:\Users\tetha\Desktop\ml-hep-off-an\experiments\2024-06-26_np_images\datafiles\agg")
    data_filepath = data_dirpath.joinpath(data_filename)

    return data_filepath


def load_agg_data(split, level):
    """
    Load the aggregated data at a particular simulation level.
    Level can be "gen" for generator level
    or "det" for detector level.

    Parameters
    ----------
    split : str
        "train" or "test"
    level : str
        "gen" for generator or "det" for detector.

    Returns
    -------
    pd.DataFrame
    """
    
    assert level in {"gen", "det"}

    filepath = make_agg_data_filepath(split)

    df = pd.read_pickle(filepath).loc[level]
    
    return df


# def scale_std(df, columns):
#     """
#     Scale specified columns of a dataframe by
#     centering at zero and setting variance to 1.

#     Return scaled dataframe.
#     """

#     df_scaled = df.copy()
   
#     for col in columns:

#         std = df_scaled[col].std()
#         mean = df_scaled[col].mean()
        
#         df_scaled[col] = (df_scaled[col] - mean) / std
   
#     return df_scaled