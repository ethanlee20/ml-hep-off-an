
"""
Utility functions.
"""

import sys
import pathlib as pl
from warnings import simplefilter

from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import uproot
import pandas as pd

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)



# File handling


# def open_tree(filepath, tree_name):
#     """
#     Open a root tree as a pandas dataframe.

#     Parameters
#     ----------
#     filepath : str
#         Root file's filepath
#     tree_name : str
#         Tree name

#     Returns
#     -------
#     pd.DataFame
#         Root tree dataframe
#     """
#     df = uproot.open(f"{filepath}:{tree_name}").arrays(library="pd")
#     return df


# def open_root(filepath):
#     """
#     Open a root file as a pandas dataframe.

#     The file can contain multiple trees.
#     Each tree will be labeled by a pandas multi-index.

#     Parameters
#     ----------
#     filepath : str
#         Root file's filepath

#     Returns
#     -------
#     pd.DataFrame
#         Root file dataframe
#     """
#     f = uproot.open(filepath)
#     tree_names = [name.split(';')[0] for name in f.keys()]
#     dfs = [f[name].arrays(library="pd") for name in tree_names] 
#     result = pd.concat(dfs, keys=tree_names)
#     return result


# def open_datafile(filepath):
#     """
#     Open a datafile as a pandas dataframe.

#     The datafile can be a root or pickled pandas dataframe file.
    
#     Parameters
#     ----------
#     filepath : str
#         Datafile's filepath
    
#     Returns
#     -------
#     pd.DataFrame
#         Datafile dataframe    
#     """
#     filepath = pl.Path(filepath)
#     assert filepath.is_file()
#     assert filepath.suffix in {".root", ".pkl"}
#     print(f"opening {filepath}")
#     if filepath.suffix == ".root":
#         return open_root(filepath) 
#     elif filepath.suffix == ".pkl":
#         return pd.read_pickle(filepath)
#     else: raise ValueError("Unknown file type.")


# def open_data_dir(dirpath):
#     """
#     Open all datafiles in a directory (recursively).

#     Return a single dataframe containing all the data.

#     Parameters
#     ----------
#     dirpath : str

#     Returns
#     -------
#     pd.DataFrame
#         Data dataframe
#     """

#     dirpath = pl.Path(dirpath)
#     assert dirpath.is_dir()
#     file_paths = list(dirpath.glob('**/*.root')) + list(dirpath.glob('**/*.pkl'))
#     dfs = [open_datafile(path) for path in file_paths]
#     if dfs == []:
#         raise ValueError("Empty dir.")
#     data = pd.concat(dfs)
#     return data


# def open_data(path):
#     """
#     Open all datafiles in a directory (if path is a directory).
#     Open the specified datafile (if path is a datafile).

#     Parameters
#     ----------
#     path : str
#         Path to data directory or datafile
    
#     Returns
#     -------
#     pd.DataFrame
#         Data dataframe
#     """

#     path = pl.Path(path)
#     if path.is_file():
#         data = open_datafile(path) 
#     elif path.is_dir(): 
#         data = open_data_dir(path) 
#     return data



# # Array handling


# def min_max(a):
#     """
#     Find the min, max (tuple) of an array or list of arrays.
    
#     Parameters
#     ----------
#     a : array or list of arrays

#     Returns
#     -------
#     min : float
#         The minimum
#     max : float
#         The maximum
#     """

#     if type(a) == list:
#         a = np.concatenate(a, axis=None)
#     min = np.nanmin(a)
#     max = np.nanmax(a)
#     return min, max   



# # Ntuple dataframe filtering


# q_squared_bounds = {
#     'all': (None, None),
#     'med': (1, 6),
#     'JPsi': (8, 11),
#     'Psi2S': (12.75, 14),
# }

    
# def section_q_squared(df, region):
#     """
#     Filter to a certain region of q^2.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Ntuple dataframe
#     region : str
#         Region of q^2
#         ('all', 'med', 'JPsi', or 'Psi2S')

#     Returns
#     -------
#     pd.DataFrame
#         Filtered dataframe

#     """
#     assert region in set(q_squared_bounds.keys())
#     return trim_df(df, 'q_squared', q_squared_bounds[region])
    

# def section(data, sig_noise=None, var=None, q_squared_split=None, gen_det=None, lim=(None, None)):
#     """
#     Apply multiple selection criteria to the data.

#     Options to select for signal or noise, a certain dataframe variable (column),
#     a region of q_squared, and generator or detector level data. 
#     If a particular variable is chosen, limits can be specified to further filter the data.
#     """

#     if gen_det == "gen":
#         data = data.loc[["gen"]]
#     elif gen_det == "det":
#         data = data.loc[["det"]]
    
#     if sig_noise == "sig":
#         data = only_signal(data)
#     elif sig_noise == "noise":
#         data = only_noise(data)

#     if q_squared_split:
#         data = section_q_squared(data, q_squared_split)

#     if var:
#         data = data[var]
#         data = trim_series(data, lim)

#     return data


# def veto_q_squared(data):
#     """
#     Veto out the JPsi and Psi2S regions in q_squared.    
#     """    
#     data = cut_df(data, 'q_squared', q_squared_bounds['JPsi'])
#     data = cut_df(data, 'q_squared', q_squared_bounds['Psi2S'])
#     return data

# def only_signal(df):
#     """
#     Filter to only signal data.

#     Return a dataframe with only signal events.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Ntuple dataframe
    
#     Returns
#     -------
#     pd.DataFrame
#         Filtered dataframe
#     """

#     isSignal = df["isSignal"]==1
#     return df[isSignal]


# def only_noise(df):
#     """
#     Filter to only noise data.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         Ntuple dataframe
    
#     Returns
#     -------
#     pd.DataFrame
#         Filtered dataframe
#     """
#     isNoise = df["isSignal"]!=1
#     return df[isNoise]


# def cut_df(d, v, s:tuple):
#     """
#     Remove elements that are in section s in variable v 
#     from dataframe d.
#     """
#     return d[~((d[v]>s[0]) & (d[v]<s[1]))]


# def trim_series(d, l:tuple):
#     """
#     Trim series d to limits l.
#     l can contain a None value to indicate an open interval.

#     Assumes that d is a pandas series.
#     """
#     if l == (None, None):
#         return d
#     elif (l[0]!=None) & (l[1]!=None):
#         return d[(d>l[0]) & (d<l[1])]
#     elif (l[0]!=None):
#         return d[d>l[0]]
#     elif (l[1]!=None):
#         return d[d<l[1]]
#     else: raise ValueError(f"Bad limits: {l}")


# def trim_df(d, v, l:tuple):
#     """
#     Trim dataframe d to limits l in variable v.
#     """
#     if l == (None, None):
#         return d
#     elif (l[0]!=None) & (l[1]!=None):
#         return d[(d[v]>l[0]) & (d[v]<l[1])]
#     elif (l[0]!=None):
#         return d[d[v]>l[0]]
#     elif (l[1]!=None):
#         return d[d[v]<l[1]]



def count_events(df):
    """
    Count the number of events in an ntuple dataframe.

    Assumes that the number of events is equal
    to the number of rows.
    
    Parameters
    ----------
    df : pd.DataFrame
        Ntuple dataframe.

    Returns
    -------
    int
        Number of events.
    """
    num_events = len(df)
    return num_events






# """Iterations"""
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



# Histogram


# def approx_num_bins(data, scale=0.2, xlim=(None,None)):
#     """
#     Approximate the number of bins for a histogram by the number of events.

#     data is assumed to be a pandas series or list of pandas series.
#     If data is a list of datasets, return the averaged best number of bins.
#     If xlim is specified, suggested number of bins is based only on data
#     from within the specified limits.
#     """

#     if type(data) == list:
#         data = [trim_series(d, xlim) for d in data]
#         return round(np.mean([np.sqrt(len(d)) for d in data])*scale)
#     return round(np.sqrt(len(trim_series(data, xlim)))*scale)   


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


# def approx_bins(data, scale=0.2, xlim=(None,None)):
#     """
#     Make bins based on the number of events.
    
#     Data is assumed to be a pandas series.
#     Returned bins are uniform size.
#     If data is a list, the bins are based on the
#     averaged suggested number of bins.
#     If xlim is specified, bins are created on the
#     interval specified by xlim.
#     If xlim is not specified, bins are created on the
#     interval given by the min and max of the given data.
#     """

#     num_bins = approx_num_bins(data, scale=scale, xlim=xlim)
#     if xlim == (None, None):
#         xlim = min_max(data)        
#     bins = make_bin_edges(start=xlim[0], stop=xlim[1], num_bins=num_bins)
#     return bins



def bin_data(df, var, num_bins, ret_edges=False):
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


# def find_bin_counts(binned):
#     """
#     Find the number of entries in each bin.

#     binned is assumed to be a pandas dataframe groupby object.
#     Return a pandas object relating each bin's interval to 
#     its count.
#     """
#     counts = binned.size()
#     return counts


# """Testing"""

# def test_df_a():
#     """
#     A test dataframe for testing purposes.
#     """

#     return pd.DataFrame(
#         {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
#         index=[3, 4, 10],
#     )


# def test_df_b():
#     """
#     A test dataframe for testing purposes.
#     """

#     return pd.DataFrame(
#         {"d": [1, 1, 2], "e": [3, 2, 1], "f": [5, 4, 7]},
#         index=[3, 4, 10],
#     )


# def test_df_c():
#     """
#     A test dataframe for testing purposes.
#     """

#     return pd.DataFrame(
#         {"d": [1, 1, 2, 7, 6, 3, 1, 4, 5, 2], "e": [3, 2, 1, 4, 10, 2, 4, 4, 5, 1]},
#         index=[3, 4, 10, 11, 13, 14, 15, 16, 18, 20]
#     )


# def test_df_d():
#     """
#     A test dataframe for testing purposes.
#     """

#     return pd.DataFrame(
#         {"d": [1, 2, 3, 2, 8, 1, 4, 5, 5, 2], "e": [3, 2, 1, 4, 10, 2, 4, 4, 5, 1]},
#         index=[3, 4, 10, 11, 13, 14, 15, 16, 18, 20]
#     )


# def test_df_vec_a():
#     """
#     A test dataframe for testing purposes.

#     Might be useful for applications requiring dataframes
#     of vectors with dimension 3.
#     """

#     return pd.DataFrame(
#         {"v_1": [2, 5, 3], "v_2": [7, 12, 1], "v_3": [1, 4, 2]},
#         index=[1, 5, 6],
#     )


# def test_df_vec_b():
#     """
#     A test dataframe for testing purposes.

#     Might be useful for applications requiring dataframes
#     of vectors with dimension 3.
#     """
    
#     return pd.DataFrame(
#         {"v_1": [4, 5, 1], "v_2": [1, 2, 5], "v_3": [7, 1, 4]},
#         index=[1, 5, 6],
#     )


# def test_df_square_mat():
#     """
#     A test dataframe for testing purposes.

#     Might be useful for applications requiring dataframes
#     of matrices with dimension 3x3.
#     """

#     return pd.DataFrame(
#         {
#             "m_11": [4, 1, 3],
#             "m_12": [6, 1, 4],
#             "m_13": [8, 5, 5],
#             "m_21": [7, 4, 2],
#             "m_22": [1, 0, 8],
#             "m_23": [7, 10, 4],
#             "m_31": [6, 4, 2],
#             "m_32": [9, 3, 2],
#             "m_33": [9, 2, 7],
#         },
#         index=[1, 5, 6],
#     )


# def test_df_4vec_a():
#     """
#     A test dataframe for testing purposes.

#     Might be useful for applications requiring dataframes
#     of vectors with dimension 4.
#     """
    
#     return pd.DataFrame(
#         {
#             "v_1": [1, 5, 4],
#             "v_2": [5, 2, 4],
#             "v_3": [8, 9, 5],
#             "v_4": [1, 5, 3],
#         },
#         index=[4, 6, 9],
#     )


# def test_df_4mom_a():
#     """
#     A test dataframe for testing purposes.

#     Might be useful for applications requiring dataframes
#     of four-momenta.
#     """
    
#     return pd.DataFrame(
#         {
#             "E": [1, 5, 4],
#             "px": [5, 2, 4],
#             "py": [8, 9, 5],
#             "pz": [1, 5, 3],
#         },
#         index=[4, 6, 9],
#     )


# def test_df_4mom_b():
#     """
#     A test dataframe for testing purposes.

#     Might be useful for applications requiring dataframes
#     of four-momenta.
#     """
    
#     return pd.DataFrame(
#         {
#             "E": [3, 2, 3],
#             "px": [1, 4, 3],
#             "py": [1, 4, 2],
#             "pz": [0, 2, 5],
#         },
#         index=[4, 6, 9],
#     )
