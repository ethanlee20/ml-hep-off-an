
from pathlib import Path
import numpy as np
import pandas as pd


from helpers import load_df, list_dc9, load_image




data_filename = "df_train.pkl"
data_dirpath = Path("../datafiles/consolidated")
data_filepath = data_dirpath.joinpath(data_filename)
df = pd.read_pickle(data_filepath).loc["gen"]

events_per_dist = 3
num_dists_per_dc9 = 3

df_by_dc9 = df.groupby("dc9")

df_sample = df_by_dc9.sample(n=events_per_dist*num_dists_per_dc9, replace=True)

df_sample_avgs = pd.DataFrame(columns=["q_squared", "costheta_mu", "costheta_K", "chi", "dc9"])

for i in range(0, len(df_sample), events_per_dist):
    df_dist = df_sample.iloc[i:i+events_per_dist]
    df_sample_avgs = pd.concat([
        None if df_sample_avgs.empty 
        else df_sample_avgs, 
        df_dist.mean().to_frame().T
    ])



breakpoint()


        