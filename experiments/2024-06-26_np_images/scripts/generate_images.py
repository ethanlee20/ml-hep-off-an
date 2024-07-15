import os.path
import sys

import pandas as pd

import lib


def make_images_dir(path):
    os.mkdir(os.path.join(path, "images_data"))


data = pd.read_pickle(sys.argv[1])
split_data = dict(
    q2_all=data,
    q2_med=data[
        (data["q_squared"] > 1) & (data["q_squared"] < 6)
    ],
)

out_dir = sys.argv[2]
make_images_dir(out_dir)

for split in split_data:
    image = lib.img_gen.generate_image_dataframe(
        split_data[split],
        ["costheta_mu", "costheta_K", "chi"],
        25,
        "q_squared",
    )
    image.to_pickle(
        os.path.join(
            out_dir, "images_data/", f"image_{split}.pkl"
        )
    )
