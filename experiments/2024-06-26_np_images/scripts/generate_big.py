
"""
Generate two dataframe files, one containing all training examples 
and the other containing all testing examples.
"""

import pandas as pd

from helpers import file_info, load_trial_range


for train in (True, False):
    
    # breakpoint()
    
    trial_range = (
        range(1, 21) if train
        else range(21, 26)
    )

    filepaths = load_trial_range(trial_range)

    df_all = pd.DataFrame(columns=["q_squared", "costheta_mu", "costheta_K", "chi", "dc9"])

    for fp in filepaths:
        
        df_to_add = pd.read_pickle(fp)
        
        df_to_add = df_to_add[["q_squared", "costheta_mu", "costheta_K", "chi"]]
        df_to_add.dropna(inplace=True)
        
        dc9 = file_info(fp)["dc9"]
        df_to_add["dc9"] = dc9

        df_all = pd.concat(
            [
                None if df_all.empty else df_all,
                df_to_add
            ]
        )

    output_filename = (
        "df_train.pkl" if train
        else "df_test.pkl"
    )

    df_all.to_pickle(f"../datafiles/consolidated/{output_filename}")


