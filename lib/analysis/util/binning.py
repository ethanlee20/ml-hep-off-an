
"""
Functions for binning data.
"""

import numpy as np
import pandas as pd


def find_bin_middles(bins):
    """
    Find the position of the middle of each bin.

    Assume uniform bin widths.
    
    Parameters
    ----------
    bins : array 
        A list of bin edges

    Returns
    -------
    array
        Bin middles
    """
    num_bins = len(bins) - 1
    bin_width = (
        np.max(bins) - np.min(bins)
    ) / num_bins
    shifted_edges = bins + 0.5 * bin_width
    return shifted_edges[:-1]

    
def make_bin_edges(start, stop, num_bins, ret_middles=False):
    """
    Make histogram bin edges.

    Include the stop edge.
    Bins are uniform size.

    Parameters
    ----------
    start : float
        The position of the first bin edge.
    stop : float
        The position of the last bin edge.
    num_bins : int
        The number of bins.
    ret_middles : bool, optional
        Whether or not to return the bin middles.

    Returns
    -------
    edges : array
        Bin edge positions.
    middles : array, optional
        Bins middles
    """
    bin_size = (stop - start) / num_bins
    edges = np.arange(start, stop + bin_size, bin_size) 
    if ret_middles:
        middles = find_bin_middles(edges)
        return edges, middles
    return edges


def bin_dataframe(df, var, num_bins, ret_edges=False):
    """
    Bin a dataframe in a particular variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be binned.
    var : str
        Name of the variable (column) to bin the dataframe in.
    num_bins : int
        Number of bins.
    ret_edges : bool, optional
        Whether or not to also return the bin edges.

    Returns
    -------
    binned : pd.api.typing.DataFrameGroupBy
        The binned dataframe.
    bin_edges : array
        Bin edge positions.
    """

    bin_edges = make_bin_edges(
        start=df[var].min(), 
        stop=df[var].max(), 
        num_bins=num_bins
    )
    bins = pd.cut(df[var], bin_edges, include_lowest=True) # the interval each event falls into
    binned = df.groupby(bins)

    if ret_edges == False:
        return binned
    return binned, bin_edges