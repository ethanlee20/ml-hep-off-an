
"""
Generate an image file (histogram style - 4D).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from ml_hep_off_an_lib.afb import afb_fn
from ml_hep_off_an_lib.s5 import calc_s5

from helpers import file_info, make_image_filename


def make_output_filepath(input_filepath, output_dirpath, level):
    
    input_filepath = Path(input_filepath)
    output_dirpath = Path(output_dirpath)

    input_file_info = file_info(input_filepath)
    output_filename = make_image_filename(
        dc9=input_file_info["dc9"], 
        trial=input_file_info["trial"], 
        level=level
    )
    
    output_filepath = output_dirpath.joinpath(output_filename)
    return output_filepath


def scale_std(a:np.array):
    stdev = np.std(a)
    mean = np.mean(a)
    scaled = (a - mean) / stdev
    return scaled


def make_hist(df_level, n_bins=4):
    bins = {
        "q_squared": np.array([0, 1, 6, 16, 20]),
        "chi": np.linspace(start=0, stop=2*np.pi, num=n_bins+1),
        "costheta_mu": np.linspace(start=-1, stop=1, num=n_bins+1),
        "costheta_K": np.linspace(start=-1, stop=1, num=n_bins+1),
    }
    
    reduce_cols = lambda df: df[list(bins.keys())]
    df_reduced = reduce_cols(df_level)
    np_reduced = df_reduced.to_numpy()
    hist, _ = np.histogramdd(np_reduced, bins=list(bins.values()), density=False)

    scaled_hist = np.zeros_like(hist)
    for channel in range(hist.shape[0]):
        scaled_hist[channel] = scale_std(hist[channel])

    return scaled_hist


def make_afb_s5_cubes(df_level, cube_shape):
    q_sq_bins = np.array([1, 17])
    df_bins = pd.cut(df_level["q_squared"], bins=q_sq_bins, include_lowest=True)
    df_grouped = df_level.groupby(df_bins, observed=False)

    df_afb = df_grouped.apply(afb_fn("mu"))
    df_s5 = df_grouped.apply(calc_s5)

    np_afb = df_afb.to_numpy()
    np_s5 = df_s5.to_numpy()

    cube_afb = np.resize(np_afb, (1, *cube_shape))
    cube_s5 = np.resize(np_s5, (1, *cube_shape))

    cubes = np.concatenate((cube_afb, cube_s5), axis=0)

    return cubes



def generate_image(input_filepath, output_dirpath, level):
    """
    level: "det" or "gen"
    """
    assert level in {"det", "gen"}
    
    input_filepath = Path(input_filepath)
    output_dirpath = Path(output_dirpath)    
    assert input_filepath.is_file()
    assert output_dirpath.is_dir()
    
    output_filepath = make_output_filepath(input_filepath, output_dirpath, level)

    df = pd.read_pickle(input_filepath)
    df = df[df["isSignal"]==1]
    df_level = df.loc[level]

    hist = make_hist(df_level)
    afb_s5 = make_afb_s5_cubes(df_level, cube_shape=hist.shape[1:])

    image = np.concatenate((hist, afb_s5), axis=0)

    np.save(output_filepath, image)
    return


def main():
    
    input_dirpath = Path("C:/Users/tetha/Desktop/ml-hep-off-an/experiments/2024-06-26_np_images/datafiles")
    output_dirpath = Path("C:/Users/tetha/Desktop/ml-hep-off-an/experiments/2024-06-26_np_images/datafiles/hist_4d_with_afb_s5")

    for input_filepath in list(input_dirpath.glob("*_re.pkl")):
        for lev in {"det", "gen"}:
            generate_image(input_filepath, output_dirpath, level=lev)

    return


if __name__ == "__main__":
    main()

