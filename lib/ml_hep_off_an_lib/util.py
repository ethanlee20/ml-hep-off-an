
"""
Utility functions.
"""


import sys
import pathlib as pl
from warnings import simplefilter

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
#import uproot
import pandas as pd

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


"""
File handling
"""

#def open_tree(file_path, tree_name):
#    """Open a root tree as a pandas dataframe."""
#
#    df = uproot.open(f"{file_path}:{tree_name}").arrays(library="pd")
#    return df


def open_root(file_path):
    """
    Open a root file as a pandas dataframe.

    The file can contain multiple trees.
    Each tree will be labeled by a pandas multi-index
    """
    f = uproot.open(file_path)
    tree_names = [name.split(';')[0] for name in f.keys()]
    dfs = [f[name].arrays(library="pd") for name in tree_names] 
    result = pd.concat(dfs, keys=tree_names)
    return result


def open_datafile(path):
    """
    Open a datafile as a pandas dataframe.

    The datafile must be a root or pickled pandas dataframe file.
    If opening a root file, tree_names must be specified.
    """
    path = pl.Path(path)
    assert path.is_file()
    assert path.suffix in {".root", ".pkl"}
    print(f"opening {path}")
    if path.suffix == ".root":
        return open_root(path) 
    elif path.suffix == ".pkl":
        return pd.read_pickle(path)
    else: raise ValueError("Unknown file type.")


def open_data_dir(path):
    """
    Open all datafiles in a directory (recursively).

    Return a single dataframe containing all the data.
    """

    path = pl.Path(path)
    assert path.is_dir()
    file_paths = list(path.glob('**/*.root')) + list(path.glob('**/*.pkl'))
    dfs = [open_datafile(path) for path in file_paths]
    if dfs == []:
        raise ValueError("Empty dir.")
    data = pd.concat(dfs)
    return data


def open_data(path):
    """
    Open all datafiles in a directory (if path is a directory).
    Open the specified datafile (if path is a datafile).
    """

    path = pl.Path(path)
    if path.is_file():
        data = open_datafile(path) 
    elif path.is_dir(): 
        data = open_data_dir(path) 
    return data



"""
Data Manipulations
"""

def cut_df(d, v, s:tuple):
    """
    Remove elements that are in section s in variable v 
    from dataframe d.
    """
    return d[~((d[v]>s[0]) & (d[v]<s[1]))]


def trim_series(d, l:tuple):
    """
    Trim series d to limits l.
    l can contain a None value to indicate an open interval.

    Assumes that d is a pandas series.
    """
    if l == (None, None):
        return d
    elif (l[0]!=None) & (l[1]!=None):
        return d[(d>l[0]) & (d<l[1])]
    elif (l[0]!=None):
        return d[d>l[0]]
    elif (l[1]!=None):
        return d[d<l[1]]
    else: raise ValueError(f"Bad limits: {l}")


def trim_df(d, v, l:tuple):
    """
    Trim dataframe d to limits l in variable v.
    """
    if l == (None, None):
        return d
    elif (l[0]!=None) & (l[1]!=None):
        return d[(d[v]>l[0]) & (d[v]<l[1])]
    elif (l[0]!=None):
        return d[d[v]>l[0]]
    elif (l[1]!=None):
        return d[d[v]<l[1]]


def min_max(a):
    """
    Find the min, max (tuple) of a, an array or list of arrays.
    """

    if type(a) == list:
        a = np.concatenate(a, axis=None)
    min = np.nanmin(a)
    max = np.nanmax(a)
    return min, max   


def count_events(df):
    """
    Count the number of events in a dataframe.

    Assumes that the number of events is equal
    to the number of rows.
    """
    num_events = len(df)
    return num_events


def only_signal(data):
    """
    Filter to only signal data.

    Return a dataframe with only signal events.
    """

    isSignal = data["isSignal"]==1
    return data[isSignal]


def only_noise(data):
    """
    Filter to only noise data.

    Return a dataframe with only noise events.
    """
    
    isNoise = data["isSignal"]!=1
    return data[isNoise]


q_squared_bounds = {
    'all': (None, None),
    'med': (1, 6),
    'JPsi': (8, 11),
    'Psi2S': (12.75, 14),
}

    
def section_q_squared(data, split):
    """
    Filter to a certain region of q squared.

    Return the data that passes the filter.
    Choices are: 'all', 'med', 'JPsi', and 'Psi2S'.
    """
    assert split in set(q_squared_bounds.keys())
    return trim_df(data, 'q_squared', q_squared_bounds[split])
    

def section(data, sig_noise=None, var=None, q_squared_split=None, gen_det=None, lim=(None, None)):
    """
    Apply multiple selection criteria to the data.

    Options to select for signal or noise, a certain dataframe variable (column),
    a region of q_squared, and generator or detector level data. 
    If a particular variable is chosen, limits can be specified to further filter the data.
    """

    if gen_det == "gen":
        data = data.loc[["gen"]]
    elif gen_det == "det":
        data = data.loc[["det"]]
    
    if sig_noise == "sig":
        data = only_signal(data)
    elif sig_noise == "noise":
        data = only_noise(data)

    if q_squared_split:
        data = section_q_squared(data, q_squared_split)

    if var:
        data = data[var]
        data = trim_series(data, lim)

    return data


def veto_q_squared(data):
    """
    Veto out the JPsi and Psi2S regions in q_squared.    
    """    
    data = cut_df(data, 'q_squared', q_squared_bounds['JPsi'])
    data = cut_df(data, 'q_squared', q_squared_bounds['Psi2S'])
    return data



"""Iterations"""
#
#def over_q_squared_splits(f):
#    """
#    Iterate a function over q squared splits.
#
#    Decorator.
#    """
#    def wrapper(*args, **kwargs):
#        for q_squared_split in q_squared_splits:
#            f(*args, q_squared_split=q_squared_split, **kwargs)
#    return wrapper



"""Histogram"""


def approx_num_bins(data, scale=0.2, xlim=(None,None)):
    """
    Approximate the number of bins for a histogram by the number of events.

    data is assumed to be a pandas series or list of pandas series.
    If data is a list of datasets, return the averaged best number of bins.
    If xlim is specified, suggested number of bins is based only on data
    from within the specified limits.
    """

    if type(data) == list:
        data = [trim_series(d, xlim) for d in data]
        return round(np.mean([np.sqrt(len(d)) for d in data])*scale)
    return round(np.sqrt(len(trim_series(data, xlim)))*scale)   


def find_bin_middles(bins):
    """
    Find the position of the middle of each bin.
    
    bins is a list of edges.
    Assumes uniform bin widths.
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

    Bins are uniform size.
    Includes the stop edge.
    """

    bin_size = (stop - start) / num_bins
    edges = np.arange(start, stop + bin_size, bin_size) 
    if ret_middles:
        middles = find_bin_middles(edges)
        return edges, middles
    return edges


def approx_bins(data, scale=0.2, xlim=(None,None)):
    """
    Make bins based on the number of events.
    
    Data is assumed to be a pandas series.
    Returned bins are uniform size.
    If data is a list, the bins are based on the
    averaged suggested number of bins.
    If xlim is specified, bins are created on the
    interval specified by xlim.
    If xlim is not specified, bins are created on the
    interval given by the min and max of the given data.
    """

    num_bins = approx_num_bins(data, scale=scale, xlim=xlim)
    if xlim == (None, None):
        xlim = min_max(data)        
    bins = make_bin_edges(start=xlim[0], stop=xlim[1], num_bins=num_bins)
    return bins


    
def bin_data(df, var, num_bins, ret_edges=False):
    """
    Bin df in intervals of variable var (equal size bins).

    i.e. Group the data by interval in var.
    If ret_edges is True, return the list of bin edge values as well.
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


def find_bin_counts(binned):
    """
    Find the number of entries in each bin.

    binned is assumed to be a pandas dataframe groupby object.
    Return a pandas object relating each bin's interval to 
    its count.
    """
    counts = binned.size()
    return counts


"""Testing"""

def test_df_a():
    """
    A test dataframe for testing purposes.
    """

    return pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
        index=[3, 4, 10],
    )


def test_df_b():
    """
    A test dataframe for testing purposes.
    """

    return pd.DataFrame(
        {"d": [1, 1, 2], "e": [3, 2, 1], "f": [5, 4, 7]},
        index=[3, 4, 10],
    )


def test_df_c():
    """
    A test dataframe for testing purposes.
    """

    return pd.DataFrame(
        {"d": [1, 1, 2, 7, 6, 3, 1, 4, 5, 2], "e": [3, 2, 1, 4, 10, 2, 4, 4, 5, 1]},
        index=[3, 4, 10, 11, 13, 14, 15, 16, 18, 20]
    )


def test_df_d():
    """
    A test dataframe for testing purposes.
    """

    return pd.DataFrame(
        {"d": [1, 2, 3, 2, 8, 1, 4, 5, 5, 2], "e": [3, 2, 1, 4, 10, 2, 4, 4, 5, 1]},
        index=[3, 4, 10, 11, 13, 14, 15, 16, 18, 20]
    )


def test_df_vec_a():
    """
    A test dataframe for testing purposes.

    Might be useful for applications requiring dataframes
    of vectors with dimension 3.
    """

    return pd.DataFrame(
        {"v_1": [2, 5, 3], "v_2": [7, 12, 1], "v_3": [1, 4, 2]},
        index=[1, 5, 6],
    )


def test_df_vec_b():
    """
    A test dataframe for testing purposes.

    Might be useful for applications requiring dataframes
    of vectors with dimension 3.
    """
    
    return pd.DataFrame(
        {"v_1": [4, 5, 1], "v_2": [1, 2, 5], "v_3": [7, 1, 4]},
        index=[1, 5, 6],
    )


def test_df_square_mat():
    """
    A test dataframe for testing purposes.

    Might be useful for applications requiring dataframes
    of matrices with dimension 3x3.
    """

    return pd.DataFrame(
        {
            "m_11": [4, 1, 3],
            "m_12": [6, 1, 4],
            "m_13": [8, 5, 5],
            "m_21": [7, 4, 2],
            "m_22": [1, 0, 8],
            "m_23": [7, 10, 4],
            "m_31": [6, 4, 2],
            "m_32": [9, 3, 2],
            "m_33": [9, 2, 7],
        },
        index=[1, 5, 6],
    )


def test_df_4vec_a():
    """
    A test dataframe for testing purposes.

    Might be useful for applications requiring dataframes
    of vectors with dimension 4.
    """
    
    return pd.DataFrame(
        {
            "v_1": [1, 5, 4],
            "v_2": [5, 2, 4],
            "v_3": [8, 9, 5],
            "v_4": [1, 5, 3],
        },
        index=[4, 6, 9],
    )


def test_df_4mom_a():
    """
    A test dataframe for testing purposes.

    Might be useful for applications requiring dataframes
    of four-momenta.
    """
    
    return pd.DataFrame(
        {
            "E": [1, 5, 4],
            "px": [5, 2, 4],
            "py": [8, 9, 5],
            "pz": [1, 5, 3],
        },
        index=[4, 6, 9],
    )


def test_df_4mom_b():
    """
    A test dataframe for testing purposes.

    Might be useful for applications requiring dataframes
    of four-momenta.
    """
    
    return pd.DataFrame(
        {
            "E": [3, 2, 3],
            "px": [1, 4, 3],
            "py": [1, 4, 2],
            "pz": [0, 2, 5],
        },
        index=[4, 6, 9],
    )
