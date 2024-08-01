
"""
Generate an image file (Shawn style).
"""

from pathlib import Path
import numpy as np
import pandas as pd

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


def generate_image(input_filepath, output_dirpath, level, n_bins=50):
    """
    level: "det" or "gen"
    """
    assert level in {"det", "gen"}
    
    input_filepath = Path(input_filepath)
    output_dirpath = Path(output_dirpath)    
    assert input_filepath.is_file()
    assert output_dirpath.is_dir()
    
    output_filepath = make_output_filepath(input_filepath, output_dirpath, level)

    bins = {
        "chi": np.linspace(start=0, stop=2*np.pi, num=n_bins+1),
        "costheta_mu": np.linspace(start=-1, stop=1, num=n_bins+1),
        "costheta_K": np.linspace(start=-1, stop=1, num=n_bins+1),
    }

    df = pd.read_pickle(input_filepath)
    df = df.loc[level]
    reduce_cols = lambda df: df[list(bins.keys()) + ["q_squared"]]
    df = reduce_cols(df)

    bin_col_names = [var + "_bin" for var in bins]

    for var, col_name in zip(bins, bin_col_names):
        df[col_name] = pd.cut(df[var], bins=bins[var], include_lowest=True)

    df_image = df.groupby(bin_col_names, observed=False).mean()["q_squared"]
    np_image = df_image.to_numpy().reshape((n_bins,)*3) # dimensions of (chi, costheta_mu, costheta_K)
    np_image_nan_is_zero = np.nan_to_num(np_image)

    np.save(output_filepath, np_image_nan_is_zero)
    return


def main():
    
    input_dirpath = Path("C:/Users/tetha/Desktop/ml-hep-off-an/experiments/2024-06-26_np_images/datafiles")
    output_dirpath = Path("C:/Users/tetha/Desktop/ml-hep-off-an/experiments/2024-06-26_np_images/datafiles/shawn_images")

    for input_filepath in list(input_dirpath.glob("*_re.pkl")):
        for lev in {"det", "gen"}:
            generate_image(input_filepath, output_dirpath, level=lev, n_bins=50)

    return


if __name__ == "__main__":
    main()

