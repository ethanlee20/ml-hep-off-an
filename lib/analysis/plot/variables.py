
from math import pi
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from matplotlib.cm import ScalarMappable

from ..util import make_bin_edges


def plot_hist(d_var, bins, color, ax, linestyle="-", label=None):
    ax.hist(d_var, bins, color=color, histtype="step", linestyle=linestyle, label=label)


def plot_variables_all(df_gen, df_det, n_bins_gen, n_bins_det, out_dir_path, alpha=0.8):
    
    variables = ["q_squared", "costheta_mu", "costheta_K", "chi"]

    x_intervals = [(0, 20), (-1, 1), (-1, 1), (0, 2*pi)]

    x_labels = [r"$q^2$ [GeV$^2$]", r"$\cos\theta_\mu$", r"$\cos\theta_K$", r"$\chi$"]

    dc9_values = df_gen["dc9"].unique()

    gen_fig, gen_axs = plt.subplots(2,2, sharey=False, layout="compressed")
    det_fig, det_axs = plt.subplots(2,2, sharey=False, layout="compressed")

    list_of_figures = (gen_fig, det_fig)
    list_of_axs = (gen_axs, det_axs)
    list_of_dfs = (df_gen, df_det)
    list_of_n_bins = (n_bins_gen, n_bins_det)
    list_of_out_file_names = ("variables_gen.png", "variables_det.png")
    
    cmap = plt.cm.coolwarm
    norm = CenteredNorm(vcenter=0, halfrange=abs(np.min(dc9_values)))

    for fig, axs, df, n_bins, file_name in zip(
        list_of_figures, 
        list_of_axs, 
        list_of_dfs, 
        list_of_n_bins, 
        list_of_out_file_names
    ):
        # breakpoint()
        bins_per_var = [make_bin_edges(*inter, n_bins) for inter in x_intervals]
        
        for var, bins, x_lab, ax in zip(variables, bins_per_var, x_labels, axs.flat): 
            
            ax.set_xlabel(x_lab)

            df_sm = df[df["dc9"]==0]
            d_var_sm = df_sm[var]
            
            # ax.text(
            #     0, 1, 
            #     r"Stdev. ($\delta C_9 = 0$) : " + f"{sm_resolution.std():.0}", 
            #     verticalalignment='bottom', 
            #     horizontalalignment='left',
            #     transform=ax.transAxes,
            # )
            
            for dc9 in dc9_values:
                color = cmap(norm(dc9), alpha=alpha)
                df_dc9 = df[df["dc9"]==dc9]
                d_var = df_dc9[var]    
                plot_hist(d_var, bins, color, ax)

            plot_hist(d_var_sm, bins, "black", ax, linestyle=(0, (1, 1)))

        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axs, orientation='vertical', label=r'$\delta C_9$')
        # breakpoint()
        fig.text(
            0, 1.02, 
            r"\textbf{Events / $\delta C_9$} : $\sim$" + f"{len(df)/len(dc9_values):.0}",
            verticalalignment='bottom', 
            horizontalalignment='left',
        )

        out_dir_path = Path(out_dir_path)
        out_file_path = out_dir_path.joinpath(file_name)
        fig.savefig(out_file_path, bbox_inches='tight')
        plt.close(fig)