
from analysis.datasets import generate_stacked_hist2d_data


out_dir_path = "../datafiles/stacked_hist2d"


generate_stacked_hist2d_data(
    "det", 
    "train", 
    num_events_per_dist=24_000, 
    num_dists_per_dc9=200, 
    out_dir_path=out_dir_path
)

generate_stacked_hist2d_data(
    "det", 
    "test", 
    num_events_per_dist=24_000, 
    num_dists_per_dc9=3, 
    out_dir_path=out_dir_path,
    replacement=False
)