
"""
A function for calculating the efficiency.
"""


import numpy as np

from analysis.util import min_max
from analysis.util import make_bin_edges



def calculate_efficiency(d_gen, d_det, n):
    """
    Calculate the efficiency.

    The efficiency of bin i is defined as the number of
    detector entries in i divided by the number of generator
    entries in i.

    The error for bin i is calculated as the squareroot of the
    number of detector entries in i divided by the number of
    generator entries in i.

    Parameters
    ----------
    d_gen : pd.Series 
        Pandas series of generator level values
    d_det : pd.Series
        Pandas series of detector level values.
    n : int
        The number of bins.

    Returns
    -------
    bin_mids : list of float
        Bin middle positions.
    eff : np.ndarray
        The efficiency per bin.
    err : np.ndarray
        The uncertainty of the efficiency per bin.
    """
    
    min, max = min_max([d_gen, d_det])
    bin_edges, bin_mids = make_bin_edges(min, max, n, ret_middles=True)

    hist_gen, _ = np.histogram(d_gen, bins=bin_edges)
    hist_det, _ = np.histogram(d_det, bins=bin_edges)

    eff = hist_det / hist_gen
    err = np.sqrt(hist_det) / hist_gen

    return bin_mids, eff, err
