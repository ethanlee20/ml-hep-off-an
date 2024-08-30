
from analysis.modeling.datasets import generate_stacked_hist2d_data


out_dir_path = "../datafiles/stacked_hist2d"


generate_stacked_hist2d_data(
    "train", 
    "gen", 
    num_events_per_dist=40_000, 
    num_dists_per_dc9=100, 
    out_dir_path=out_dir_path
)

generate_stacked_hist2d_data(
    "test", 
    "gen", 
    num_events_per_dist=40_000, 
    num_dists_per_dc9=20, 
    out_dir_path=out_dir_path
)