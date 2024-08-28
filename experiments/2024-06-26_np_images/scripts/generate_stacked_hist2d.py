
from analysis.modeling.datasets import generate_stacked_hist2d_data


num_events_per_dist = 40_000
num_dists_per_dc9 = 10
out_dir_path = "../datafiles/stacked_hist2d"

for split in {"train", "test"}:
    for level in {"gen", "det"}:
        generate_stacked_hist2d_data(
            split, 
            level, 
            num_events_per_dist, 
            num_dists_per_dc9, 
            out_dir_path
        )

