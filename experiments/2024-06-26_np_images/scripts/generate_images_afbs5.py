
"""
Generate an image file (of afb and s5 values).
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
    df = df.loc[level]
    df = df[df["isSignal"]==1]

    q_sq_bins = np.array([0, 1, 6, 16, 20])

    df_bins = pd.cut(df["q_squared"], bins=q_sq_bins, include_lowest=True)
    df_grouped = df.groupby(df_bins, observed=False)
    df_image_afb = df_grouped.apply(afb_fn("mu"))
    df_image_s5 = df_grouped.apply(calc_s5)
    np_image_afb = np.expand_dims(df_image_afb.to_numpy(), axis=0)
    np_image_s5 = np.expand_dims(df_image_s5.to_numpy(), axis=0)
    np_image = np.concatenate((np_image_afb, np_image_s5), axis=0)
    assert ~(np.isnan(np_image).any()), "Uh oh has a Nan." 
    
    np.save(output_filepath, np_image)
    return


def main():
    
    input_dirpath = Path("C:/Users/tetha/Desktop/ml-hep-off-an/experiments/2024-06-26_np_images/datafiles")
    output_dirpath = Path("C:/Users/tetha/Desktop/ml-hep-off-an/experiments/2024-06-26_np_images/datafiles/ensemble")

    for input_filepath in list(input_dirpath.glob("*_re.pkl")):
        for lev in {"det", "gen"}:
            generate_image(input_filepath, output_dirpath, level=lev)

    return


if __name__ == "__main__":
    main()

