
from analysis.modeling.datasets import generate_gnn_input_data

out_dir_path = "../datafiles/gnn_input"
level = "gen"
num_events_per_dist = 100
num_dists_per_dc9 = 5

for split in {"train", "test"}:
    generate_gnn_input_data(level, split, num_events_per_dist, num_dists_per_dc9, out_dir_path)