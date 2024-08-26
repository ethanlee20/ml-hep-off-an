
"""
Generate an image file (ensemble of afb, s5, and cube).
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from analysis.afb import afb_fn
from analysis.s5 import calc_s5

from helpers import file_info, make_image_filename


def make_output_filepath(input_filepath, output_dirpath, ext, level):
    
    input_filepath = Path(input_filepath)
    output_dirpath = Path(output_dirpath)

    input_file_info = file_info(input_filepath)
    output_filename = make_image_filename(
        dc9=input_file_info["dc9"], 
        trial=input_file_info["trial"], 
        ext=ext,
        level=level,
    )
    
    output_filepath = output_dirpath.joinpath(output_filename)
    return output_filepath


def make_afb_s5_array(df_level):
    """
    First row is afb, second row is s5
    """

    q_sq_bins = np.array([0, 1, 6, 16, 20])

    df_bins = pd.cut(df_level["q_squared"], bins=q_sq_bins, include_lowest=True)
    df_grouped = df_level.groupby(df_bins, observed=False)
    df_image_afb = df_grouped.apply(afb_fn("mu"))
    df_image_s5 = df_grouped.apply(calc_s5)
    np_image_afb = np.expand_dims(df_image_afb.to_numpy(), axis=0)
    np_image_s5 = np.expand_dims(df_image_s5.to_numpy(), axis=0)
    np_image = np.concatenate((np_image_afb, np_image_s5), axis=0)
    assert ~(np.isnan(np_image).any()), "Uh oh has a Nan." 
    
    return np_image


def make_cube(df_level, n_bins=5):
    bins = {
        "q_squared": np.array([0, 1, 6, 16, 20]),
        "chi": np.linspace(start=0, stop=2*np.pi, num=n_bins+1),
        "costheta_mu": np.linspace(start=-1, stop=1, num=n_bins+1),
        "costheta_K": np.linspace(start=-1, stop=1, num=n_bins+1),
    }

    reduce_cols = lambda df: df[list(bins.keys())]
    df_level = reduce_cols(df_level)

    hist, _ = np.histogramdd(df_level.to_numpy(), bins=list(bins.values()), density=False)
    
    return hist


def generate_ensemble_image(input_filepath, output_dirpath, level):
    assert level in {"det", "gen"}

    input_filepath = Path(input_filepath)
    output_dirpath = Path(output_dirpath)    
    assert input_filepath.is_file()
    assert output_dirpath.is_dir()
    
    ext = ".img"
    output_filepath = make_output_filepath(input_filepath, output_dirpath, ext, level)

    df = pd.read_pickle(input_filepath)
    df_level = df.loc[level]
    df_level = df[df["isSignal"]==1]

    image = {"afb_s5":make_afb_s5_array(df_level), "cube":make_cube(df_level)}

    with open(output_filepath, "wb") as f:
        pickle.dump(image, f)
    return


def main():
    
    input_dirpath = Path("C:/Users/tetha/Desktop/ml-hep-off-an/experiments/2024-06-26_np_images/datafiles")
    output_dirpath = Path("C:/Users/tetha/Desktop/ml-hep-off-an/experiments/2024-06-26_np_images/datafiles/ensemble")

    for input_filepath in list(input_dirpath.glob("*_re.pkl")):
        for lev in {"det", "gen"}:
            generate_ensemble_image(input_filepath, output_dirpath, level=lev)

    return


if __name__ == "__main__":
    main()

