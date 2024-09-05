
from pathlib import Path

import pandas as pd


def scale_std_df(df, columns):
    """
    Scale specified columns of a dataframe by
    centering at zero and setting variance to 1.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str

    Returns
    -------
    pd.DataFrame    
    """

    df_scaled = df.copy()
   
    for col in columns:

        std = df_scaled[col].std()
        mean = df_scaled[col].mean()
        
        df_scaled[col] = (df_scaled[col] - mean) / std
   
    return df_scaled


def make_agg_data_filepath(level, split):
    """
    Create the filepath of the aggregated data file.

    Parameters
    ----------
    split : str
        "train" gives the training data filepath.
        "test" gives the testing data filepath.
    level : str
        "gen" for generator or "det" for detector.

    Returns
    -------
    pathlib.Path
    """

    assert level in {"gen", "det"}
    assert split in {"train", "test"}

    data_filename = f"df_{level}_{split}.pkl"
    data_dirpath = Path(r"C:\Users\tetha\Desktop\ml-hep-off-an\experiments\2024-06-26_np_images\datafiles\agg")
    data_filepath = data_dirpath.joinpath(data_filename)

    return data_filepath


def load_agg_data(level, split):
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

    filepath = make_agg_data_filepath(level, split)

    df = pd.read_pickle(filepath)
    
    return df




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


def find_raw_data_paths(trial_range:range, raw_data_dir_path="../datafiles/raw"):
    """
    Generate a list of raw datafile paths for given trials.

    Parameters
    ----------
    trial_range : range
        Range of trial numbers.
    raw_data_dir_path : str, optional
        The path of the raw datafiles directory.

    Returns
    -------
    list of pathlib.Path
    """
    raw_data_dir_path = Path(raw_data_dir_path)

    filepaths = []
    for t in trial_range:
        filepaths += raw_data_dir_path.glob(f"dc9_*_{t}_re.pkl")

    return filepaths