
"""
Generate two dataframe files, one containing all training examples 
and the other containing all testing examples.
"""

import pandas as pd

from ..helpers import file_info, find_raw_data_paths, make_agg_data_filepath

 
def generate_agg_data(split, features:list):
    """
    Generate datafiles aggregated over all trials.

    Parameters
    ----------
    split : str
        "train" or "test"
    
    Side Effects
    ------------
    - Save aggregated datafiles 
        (one for the generator level data and
        one for the detector level data).
    """
    trial_range = (
        range(1, 31) if split=="train"
        else range(31, 41)
    )

    filepaths = find_raw_data_paths(trial_range)

    df_agg = pd.DataFrame(
        columns=(features + labels)
    )

    for fp in filepaths:
        
        df_trial = pd.read_pickle(fp)
        
        df_trial = df_trial[features]
        df_trial.dropna(inplace=True)
        
        dc9 = file_info(fp)["dc9"]
        df_trial["dc9"] = dc9

        df_agg = pd.concat(
            [
                None if df_agg.empty else df_agg,
                df_trial
            ]
        )

    output_filename = (
        "df_train.pkl" if train
        else "df_test.pkl"
    )

    df_agg.to_pickle(f"../datafiles/agg/{output_filename}")


