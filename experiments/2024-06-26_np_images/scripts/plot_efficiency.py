
from pathlib import Path
import pandas as pd

from analysis.plot import plot_efficiency_all, setup_mpl_params


setup_mpl_params()

agg_data_dir = Path("../datafiles/agg")

data_gen_filename = "df_gen_train.pkl"
data_det_filename = "df_det_train.pkl"

data_gen_filepath = agg_data_dir.joinpath(data_gen_filename) 
data_det_filepath = agg_data_dir.joinpath(data_det_filename)

df_gen = pd.read_pickle(data_gen_filepath)
df_det = pd.read_pickle(data_det_filepath)



plot_efficiency_all(df_gen, df_det, "../plots", n=25)