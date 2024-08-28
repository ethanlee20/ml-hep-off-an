
import numpy as np

from torch import from_numpy, tensor
from torch.utils.data import Dataset

from .helpers import load_agg_data, bootstrap


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

    n_bins = 5
    bins = {
        "q_squared": np.array([0, 1, 6, 12, 16, 20]),
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


class Stacked_Hist2d_Dataset(Dataset):
    """
    Dataset of stacked 2d histograms.
    """
    def __init__(self, level:str, train:bool, num_events_per_dist, num_dists_per_dc9, normalize=True):

        agg_data = load_agg_data(train, level)

        bootstrapped_data = bootstrap(agg_data, num_events_per_dist, num_dists_per_dc9)

        labels = [d.iloc[0]["dc9"] for d in bootstrapped_data]

        hists = [make_hist_stack(d, normalize=normalize) for d in bootstrapped_data]

        self.y = labels
        self.x = hists

    def __len__(self):

        return len(self.x)

    def __getitem__(self, idx):
        y_torch = tensor([self.y[idx]])
        x_torch = from_numpy(self.x[idx])
        return x_torch, y_torch
    