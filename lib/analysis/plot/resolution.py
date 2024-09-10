

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from matplotlib.cm import ScalarMappable

from ..calc import calculate_resolution
from ..util import make_bin_edges


def plot_resolution_curve(d_recon, d_truth, bins, color, ax, periodic=False, label=None, linestyle="-"):

    """
    Plot the resolution of a particular data.

    Parameters
    ----------
    d_recon : pd.Series
        Reconstructed data pandas series.
    d_truth : pd.Series
        MC truth data pandas series.
    bins : np.ndarray
        Bin edges.
    color : matplotlib color
        The color of the curve.
    ax : mpl.axes.Axes
        The axes to plot on.
    periodic : bool, optional
        Whether or not to apply periodicity in the resolution calculation.

    Side Effects
    ------------
    - Plots the resolution on the given axes.
    """    

    resolution = calculate_resolution(d_recon, d_truth, periodic)

    ax.hist(resolution, bins=bins, density=True, histtype="step", color=color, label=label, linestyle=linestyle)


def plot_resolution_all(df_det, out_dir_path, n_bins=10, alpha=0.8):
    """
    Plot a histogram of the resolution for variables of interest 
    for all delta C9 values.

    Parameters
    ----------
    df_det : pd.DataFrame 
        The aggregated detector level dataframe.
    n_bins : int
        The number of bins per plot.
    out_dir_path : str
        The path of the directory to which to save the plot.
    alpha : float, optional
        The curve transparency.
    
    Side Effects
    ------------
    - Saves a plot to the specified directory.
    """

    variables = ["q_squared", "costheta_mu", "costheta_K", "chi"]

    x_intervals = [(-0.05, 0.05), (-0.01, 0.01), (-0.02, 0.02), (-0.02, 0.02)]

    bins_per_var = [make_bin_edges(*inter, n_bins) for inter in x_intervals]

    x_labels = [r"$R_{q^2}$ [GeV$^2$]", r"$R_{\cos\theta_\mu}$", r"$R_{\cos\theta_K}$", r"$R_{\chi}$"]

    dc9_values = df_det["dc9"].unique()

    fig, axs = plt.subplots(2, 2, sharey=False, layout="compressed")

    cmap = plt.cm.coolwarm
    norm = CenteredNorm(vcenter=0, halfrange=abs(np.min(dc9_values)))

    for var, bins, x_lab, ax in zip(variables, bins_per_var, x_labels, axs.flat): 
        
        if var == "chi":
            periodic = True
        else: periodic = False

        ax.set_xlabel(x_lab)

        df_sm = df_det[df_det["dc9"]==0]
        d_sm_recon = df_sm[var]
        d_sm_truth = df_sm[var+"_mc"]
        
        sm_resolution = calculate_resolution(d_sm_recon, d_sm_truth)
        ax.text(
            0, 1, 
            r"Stdev. ($\delta C_9 = 0$) : " + f"{sm_resolution.std():.0}", 
            verticalalignment='bottom', 
            horizontalalignment='left',
            transform=ax.transAxes,
        )
        
        for dc9 in dc9_values:
            color = cmap(norm(dc9), alpha=alpha)
            df_dc9 = df_det[df_det["dc9"]==dc9]
            d_recon = df_dc9[var]    
            d_truth = df_dc9[var+"_mc"]    
            plot_resolution_curve(d_recon, d_truth, bins, color, ax, periodic)

        plot_resolution_curve(d_sm_recon, d_sm_truth, bins, "dimgrey", ax, periodic, label=r"SM" + "\n" + r"($\delta C_9 = 0$)", linestyle=(0, (1, 1)))

    axs.flat[0].legend()
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axs, orientation='vertical', label=r'$\delta C_9$')

    fig.text(
        0, 1.02, 
        r"\textbf{Reconstructed Events / $\delta C_9$} : $\sim$" + f"{len(df_det)/len(dc9_values):.0}",
        verticalalignment='bottom', 
        horizontalalignment='left',
    )

    fig.text(
        0.82, 1.02,  
        r"\textbf{Normalized}",
        verticalalignment='bottom', 
        horizontalalignment='right',
    )

    out_dir_path = Path(out_dir_path)
    out_file_path = out_dir_path.joinpath("resolution.png")
    plt.savefig(out_file_path, bbox_inches='tight')
    plt.close(fig)



