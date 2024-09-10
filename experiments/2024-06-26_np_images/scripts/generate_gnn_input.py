
from analysis.datasets import generate_gnn_input_data

out_dir_path = "../datafiles/gnn_input"
level = "det"
num_events_per_dist = 40_000

generate_gnn_input_data(level, "train", num_events_per_dist=24_000, num_dists_per_dc9=50, out_dir_path=out_dir_path)
generate_gnn_input_data(level, "test", num_events_per_dist=24_000, num_dists_per_dc9=3, out_dir_path=out_dir_path, replacement=False)