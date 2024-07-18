
"""
Generate an image file.
"""

from math import pi
from pathlib import Path
import numpy as np
import pandas as pd
from ml_hep_off_an_lib.util import make_bin_edges
from helpers import file_info


def make_image_filename(dc9, trial, level):
    name = f"dc9_{dc9}_{trial}_img_{level}.npy"
    return name


def make_edges_filename(dc9, trial, level):
    name = f"dc9_{dc9}_{trial}_edges_{level}.npy"
    return name


def make_bins(domains, n_bins=10):
    """domains - list of tuples i.e. (min, max)"""
    bins = [make_bin_edges(*dom, n_bins) for dom in domains]
    return bins


def datafile_to_images(filepath, ell='mu'):
    filepath = Path(filepath)
    data_dir = filepath.parent
    info = file_info(filepath)
    
    vars = [f"costheta_{ell}", "costheta_K", "chi"] 
    domains = [(-1, 1), (-1, 1), (0, 2*pi)]
    bins = make_bins(domains)

    df = pd.read_pickle(filepath)
    df = df[df["isSignal"]==1]
    df = df[vars]

    levels = ["gen", "det"]
    for lev in levels:
        df_level = df.loc[lev]
        
        hist, edges = np.histogramdd(df_level.to_numpy(), bins=bins, density=False)
        edges = np.array(edges) 

        image_filename = make_image_filename(info["dc9"], info["trial"], lev)
        edges_filename = make_edges_filename(info["dc9"], info["trial"], lev)
        np.save(data_dir.joinpath(image_filename), hist)
        np.save(data_dir.joinpath(edges_filename), edges)


def main():
    for filepath in ["../datafiles/dc9_-0.08_1_re.pkl"]: #list(data_dir.glob("*_re.pkl")):
        datafile_to_images(filepath)

        
    

if __name__ == "__main__":
    main() 