
from math import pi
from pathlib import Path
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.transforms import Bbox

from ml_hep_off_an_lib.plot import setup_mpl_params, plot_volume_slices

from helpers import file_info, list_dc9, load_image_all_trials

setup_mpl_params()


def make_title(ell, level, n_events, dc9):
    assert ell in {"mu", "e"}
    assert level in {"gen", "det"}
    
    if ell == "mu":
        ell = r"$\mu$"
    elif ell == "e":
        ell = r"$e$"
    
    if level == "gen":
        level = "generator"
    elif level == "det":
        level = "detector"
    
    result = ell + f", {level}" + '\n' + f"{n_events:g} events" +'\n' + r"$\delta C_9 =$" + f"${dc9}$"
    return result
    

for level in ["gen", "det"]:
    cmap = plt.cm.magma
    norm = Normalize()
    
    for i, dc9 in enumerate(list_dc9()):
        
        image = load_image_all_trials(dc9, level)
        n_events = np.sum(image)

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        plot_volume_slices(ax, image, cmap, norm)

        ax.set_xlabel(r"$\cos\theta_\ell$", labelpad=0)
        ax.set_ylabel(r"$\cos\theta_K$", labelpad=0)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(r"$\chi$", labelpad=-3, rotation="horizontal") 

        n_ticks = 3
        costheta_ell_ticks = [f"{t:.2g}" for t in np.linspace(-1, 1, n_ticks).tolist()]
        costheta_k_ticks = [f"{t:.2g}" for t in np.linspace(-1, 1, n_ticks).tolist()]
        chi_ticks = ['0', r"$\pi$", r"$2\pi$"]
        assert len(chi_ticks) == n_ticks
        ax.set_xticks(np.linspace(0, 9, n_ticks), costheta_ell_ticks)
        ax.set_yticks(np.linspace(0, 9, n_ticks), costheta_k_ticks)
        ax.set_zticks(np.linspace(0, 9, n_ticks), chi_ticks)
        ax.tick_params(pad=0.5)

        title = make_title("mu", level, n_events, dc9)
        ax.set_title(title, loc="right", y=0.95, pad=-14)
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, location="left", shrink=0.5, pad=0.02)
        cbar.set_label("Counts", size=11)

        bbox = Bbox([[0.7,0.2], [5.1,3.8]])
        plt.savefig(f"../plots/image_slices_{level}/{i:02}_image_slices_dc9_{dc9}_{level}_all_trials.png", bbox_inches=bbox)
        plt.close(fig)
