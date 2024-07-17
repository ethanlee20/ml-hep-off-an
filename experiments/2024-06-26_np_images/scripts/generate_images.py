"""
Generate an image file.
"""

from math import pi
from pathlib import Path
import numpy as np
import pandas as pd
from ml_hep_off_an_lib.util import make_bin_edges
from helpers import file_info


def main():
    ell = "mu"
    n_bins = 50

    vars = [f"costheta_{ell}", "costheta_K", "chi"] 
    domains = [(-1, 1), (-1, 1), (0, 2*pi)]
    bins = [make_bin_edges(*dom, n_bins) for dom in domains]

    data_dir = Path("../datafiles")
    for filepath in ["../datafiles/dc9_-0.08_1_re.pkl"]: #list(data_dir.glob("*_re.pkl")):
        df = pd.read_pickle(filepath)
        info = file_info(filepath)
        df = df[df["isSignal"]==1]
        df = df[vars]
        for level in ["gen", "det"]:
            df_level = df.loc[level]
            hist, edges = np.histogramdd(df_level.to_numpy(), bins=bins)
            np.save(data_dir.joinpath(f"dc9_{info["dc9"]}_{info["trial"]}_img_{level}.npy"), hist)
            np.save(data_dir.joinpath(f"dc9_{info["dc9"]}_{info["trial"]}_edges_{level}.npy"), np.array(edges))


if __name__ == "__main__":
    main() 


# def make_images_dir(path):
#     os.mkdir(os.path.join(path, "images_data"))


# data = pd.read_pickle(sys.argv[1])
# split_data = dict(
#     q2_all=data,
#     q2_med=data[
#         (data["q_squared"] > 1) & (data["q_squared"] < 6)
#     ],
# )

# out_dir = sys.argv[2]
# make_images_dir(out_dir)

# for split in split_data:
#     image = lib.img_gen.generate_image_dataframe(
#         split_data[split],
#         ["costheta_mu", "costheta_K", "chi"],
#         25,
#         "q_squared",
#     )
#     image.to_pickle(
#         os.path.join(
#             out_dir, "images_data/", f"image_{split}.pkl"
#         )
#     )
