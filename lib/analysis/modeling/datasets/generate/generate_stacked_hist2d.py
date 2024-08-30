
"""
Functions to generate datafiles needed to construct datasets.
"""

from pathlib import Path

import numpy as np

from ..helpers import load_agg_data, bootstrap


def std_scale(arr):
    """
    Transform an array by shifting its mean to zero
    and scaling its standard deviation to one.

    Operate on the entire array.

    Parameters
    ----------
    arr : np.ndarray
        The array to transform.

    Returns
    -------
    np.ndarray
        The transformed array.
    """

    mean = np.mean(arr)
    stdev = np.std(arr)

    arr_shifted = arr - mean
    arr_shifted_scaled = arr_shifted / stdev
    
    return arr_shifted_scaled


def make_hist_stack(df, normalize=True):
    """
    Make a stack of two dimensional histograms
    of all combinations of input variables.

    The final array is three dimensional.

    Parameters
    ----------
    df : pd.DataFrame
        Ntuple dataframe.
    normalize : bool, optional
        Whether or not to standard scale
        each 2d histogram.
    Returns
    -------
    numpy.ndarray
    """

    n_bins = 20
    bins = {
        "q_squared": np.linspace(start=0, stop=20, num=n_bins+1),
        "chi": np.linspace(start=0, stop=2*np.pi, num=n_bins+1),
        "costheta_mu": np.linspace(start=-1, stop=1, num=n_bins+1),
        "costheta_K": np.linspace(start=-1, stop=1, num=n_bins+1),
    }

    combinations = [
        ["q_squared", "costheta_mu"],
        ["q_squared", "costheta_K"],
        ["q_squared", "chi"],
        ["costheta_K", "chi"],
        ["costheta_mu", "chi"],
        ["costheta_K", "costheta_mu"]
    ]

    hists = [ 
        np.histogram2d(df[c[0]], df[c[1]], bins=(bins[c[0]], bins[c[1]]))[0]
        for c in combinations
    ]

    if normalize:
        hists = [std_scale(h) for h in hists]

    hists_expanded = [np.expand_dims(h, axis=0) for h in hists]
    
    hist_stack = np.concatenate(hists_expanded, axis=0)

    return hist_stack


def generate_stacked_hist2d_data(split, level, num_events_per_dist, num_dists_per_dc9, out_dir_path, normalize=True):
    """
    Generate stacked 2d histograms from data bootstrapped
    from original aggregated ntuple data.

    Parameters
    ----------
    train: str
    level: str
    num_events_per_dist: int
    num_dists_per_dc9: int

    Side Effects
    ------------
    - Saves to disk a file containing the stacked histogram array 
        and a file containing the corresponding labels.
    """
    agg_data = load_agg_data(split, level)

    bootstrapped_data = bootstrap(agg_data, num_events_per_dist, num_dists_per_dc9)

    labels = [d.iloc[0]["dc9"] for d in bootstrapped_data]
    labels_np = np.array(labels)

    hists_list = [make_hist_stack(d, normalize=normalize) for d in bootstrapped_data]

    hists_list_expanded = [np.expand_dims(h, axis=0) for h in hists_list]

    hists_np = np.concat(hists_list_expanded, axis=0)

    out_dir_path = Path(out_dir_path)
    assert out_dir_path.is_dir()

    out_file_base_name = f"{level}_{split}"
    out_hist_filename = out_file_base_name + "_hist"
    out_label_filename = out_file_base_name + "_label"
    
    out_hist_filepath = out_dir_path.joinpath(out_hist_filename)
    out_label_filepath = out_dir_path.joinpath(out_label_filename)

    np.save(out_hist_filepath, hists_np)
    np.save(out_label_filepath, labels_np)


