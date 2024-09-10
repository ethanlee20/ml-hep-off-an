
from math import pi
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from matplotlib.cm import ScalarMappable

from analysis.calc import calculate_efficiency


def plot_efficiency_curve(d_gen, d_det, color, ax, interval, n=10, size=2, plot_error=False, label=None):
    """
    Plot the efficiency of a particular data.

    Parameters
    ----------
    df_gen : pd.DataFrame
        The aggregated generator level dataframe.
    df_det : pd.DataFrame
        The aggregated detector level dataframe.
    variable : str
        The name of the variable of which to plot the efficiency.
    dc9 : float
        The delta C9 value of the data.
    ax : mpl.axes.Axes
        The axes on which to plot.
    cmap :
        Matplotlib colormap
    norm :
        Normalization for colormap.
    n : int, optional
        The number of datapoints to plot.
    alpha : float, optional
        The transparency of the points.

    Side Effects
    ------------
    - Plots the efficiency on the specified axes.
    """

    x, y, err = calculate_efficiency(d_gen, d_det, interval, n)
    if not plot_error:
        elinewidth = 0
    else: elinewidth = 0.5
    ax.errorbar(x, y, yerr=10*err, fmt='o', color=color, ecolor=color, elinewidth=elinewidth, capsize=0, markersize=size, label=label)


def plot_efficiency_all(df_gen, df_det, out_dir_path, n=10, alpha=0.8):
    """
    Plot the efficiency for variables of interest and for all delta C9 values.

    Parameters
    ----------
    df_gen : pd.DataFrame
        The aggregated dataframe.
    df_det : pd.DataFrame
        The aggregated dataframe.
    n : int, optional
        The number of datapoints to plot per curve.
    alpha : float, optional
        The transparency of the datapoints.

    Side Effects
    ------------
    - Saves a plot to the specified directory.
    """
    
    variables = ["q_squared", "costheta_mu", "costheta_K", "chi"]

    x_intervals = [(0, 20), (-1, 1), (-1, 1), (0, 2*pi)]

    x_labels = [r"$q^2$ [GeV$^2$]", r"$\cos\theta_\mu$", r"$\cos\theta_K$", r"$\chi$"]
    
    dc9_values = df_gen["dc9"].unique()

    fig, axs = plt.subplots(2,2, sharey=True, layout="compressed")

    cmap = plt.cm.coolwarm
    norm = CenteredNorm(vcenter=0, halfrange=abs(np.min(dc9_values)))

    for var, inter, x_lab, ax in zip(variables, x_intervals, x_labels, axs.flat): 
        
        ax.set_xlabel(x_lab)
        ax.set_ylim(bottom=0, top=0.64)

        if var == "chi":
            ax.set_xticks([0, pi, 2*pi])
            ax.set_xticklabels([r"$0$",  r"$\pi$", r"$2\pi$"])
        
        for dc9 in dc9_values:
            color = cmap(norm(dc9), alpha=alpha)
            d_gen = df_gen[df_gen["dc9"]==dc9][var]    
            d_det = df_det[df_det["dc9"]==dc9][var]    
            plot_efficiency_curve(d_gen, d_det, color, ax, inter, n)

        d_sm_gen = df_gen[df_gen["dc9"]==0][var]
        d_sm_det = df_det[df_det["dc9"]==0][var]
        plot_efficiency_curve(d_sm_gen, d_sm_det, "dimgrey", ax, inter , n, size=1, plot_error=True, label=r"SM ($\delta C_9 = 0$)")

    axs.flat[0].legend()
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axs, orientation='vertical', label=r'$\delta C_9$')
    fig.supylabel(r"$\varepsilon$")

    fig.text(
        0, 1.02,  
        r"\textbf{Generator Events / $\delta C_9$} : $\sim$" + f"{len(df_gen)/len(dc9_values):.0}\n" +
        r"\textbf{Reconstructed Events / $\delta C_9$} : $\sim$" + f"{len(df_det)/len(dc9_values):.0}",
        verticalalignment='bottom', 
        horizontalalignment='left',
    )

    fig.text(
        0.82, 1.02,  
        r"\textbf{SM Errorbars $\times 10$}",
        verticalalignment='bottom', 
        horizontalalignment='right',
    )

    out_dir_path = Path(out_dir_path)
    out_file_path = out_dir_path.joinpath("efficiency.png")
    plt.savefig(out_file_path, bbox_inches='tight')
    plt.close(fig)

    
