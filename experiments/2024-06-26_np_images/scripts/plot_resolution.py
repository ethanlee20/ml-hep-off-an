
from pathlib import Path
import pandas as pd

from analysis.plot import plot_resolution_all, setup_mpl_params


setup_mpl_params()

agg_data_dir = Path("../datafiles/agg")

data_det_filename = "df_det_train.pkl"

data_det_filepath = agg_data_dir.joinpath(data_det_filename)

df_det = pd.read_pickle(data_det_filepath)

plot_resolution_all(df_det, "../plots", n_bins=200)