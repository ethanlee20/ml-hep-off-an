
from pathlib import Path

import numpy as np

from ..helpers import load_agg_data, sample_sets, scale_std_df

from ...calc import calc_afb_of_q_squared, calc_s5_of_q_squared


def generate_gnn_input_data(level, split, num_events_per_dist, num_dists_per_dc9, out_dir_path, normalize=True, replacement=True):
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
    
    base_features = ["q_squared", "costheta_mu", "costheta_K", "chi"]

    df_agg = load_agg_data(level, split)
    
    if normalize:
        df_agg = scale_std_df(df_agg, base_features)
    
    boots = sample_sets(df_agg, num_events_per_dist, num_dists_per_dc9, replacement)

    labels = [d.iloc[0]["dc9"] for d in boots]
    labels_np = np.array(labels)

    feature_dfs = [d.drop(columns="dc9") for d in boots]

    for df in feature_dfs:
        df["q_squared_avg"] = df["q_squared"].mean()
        df["costheta_mu_avg"] = df["costheta_mu"].mean()
        df["afb"] = calc_afb_of_q_squared(df, "mu", 1)[1].values.item()
        df["s5"] = calc_s5_of_q_squared(df, 1)[1].values.item()
        # breakpoint()

    feature_np_arrays = [d.to_numpy() for d in feature_dfs]
    feature_np_arrays_expanded = [np.expand_dims(d, axis=0) for d in feature_np_arrays]
    feature_np_array_agg = np.concat(feature_np_arrays_expanded, axis=0)

    out_dir_path = Path(out_dir_path)
    assert out_dir_path.is_dir()    

    out_file_base_name = f"{level}_{split}"
    out_feature_filename = out_file_base_name + "_feat"
    out_label_filename = out_file_base_name + "_label"
    
    out_feature_filepath = out_dir_path.joinpath(out_feature_filename)
    out_label_filepath = out_dir_path.joinpath(out_label_filename)

    np.save(out_feature_filepath, feature_np_array_agg)
    np.save(out_label_filepath, labels_np)



