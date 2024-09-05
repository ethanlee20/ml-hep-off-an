
from pathlib import Path

import numpy as np

from ..helpers import load_agg_data, bootstrap, scale_std_df


def generate_gnn_input_data(level, split, num_events_per_dist, num_dists_per_dc9, out_dir_path, normalize=True):
    """
    Generate and save a numpy array of bootstrapped data and a corresponding array of labels.

    Parameters
    ----------
    level
    split
    num_events_per_dist
    num_dists_per_dc9
    out_dir_path
    normalize

    Side Effects
    ------------
    - Save a file containing the features and a file containing the labels.
    """
    
    df_agg = load_agg_data(level, split)
    boots = bootstrap(df_agg, num_events_per_dist, num_dists_per_dc9)

    labels = [d.iloc[0]["dc9"] for d in boots]
    labels_np = np.array(labels)

    features = [d.drop(columns="dc9") for d in boots]

    if normalize:
        features = [scale_std_df(d, d.columns.to_list()) for d in features]

    features_np_list = [d.to_numpy() for d in features]
    features_np_list_expanded = [np.expand_dims(d, axis=0) for d in features_np_list]
    features_np = np.concat(features_np_list_expanded, axis=0)

    out_dir_path = Path(out_dir_path)
    assert out_dir_path.is_dir()    

    out_file_base_name = f"{level}_{split}"
    out_feature_filename = out_file_base_name + "_feat"
    out_label_filename = out_file_base_name + "_label"
    
    out_feature_filepath = out_dir_path.joinpath(out_feature_filename)
    out_label_filepath = out_dir_path.joinpath(out_label_filename)

    np.save(out_feature_filepath, features_np)
    np.save(out_label_filepath, labels_np)



