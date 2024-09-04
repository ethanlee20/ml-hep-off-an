
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from matplotlib.cm import ScalarMappable

from analysis.calc import calculate_efficiency


def plot_efficiency_curve(df_data, variable, dc9, ax, cmap, norm, n=10, alpha=0.8):
    """
    Plot the efficiency of a particular data.

    Parameters
    ----------
    df_data : pd.DataFrame
        The aggregated dataframe.
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

    d_generator = df_data[variable+"_mc"]
    d_detector = df_data[variable]
    x, y, err = calculate_efficiency(d_generator, d_detector, n)
    color = cmap(norm(dc9), alpha=alpha)
    ax.errorbar(x, y, yerr=err, fmt='none', ecolor=color, elinewidth=0.5, capsize=0)
    ax.scatter(x, y, s=4, color=color)
    return


def plot_efficiency_all(df_data, out_dir_path, n=10, alpha=0.8):
    """
    Plot the efficiency for variables of interest and for all delta C9 values.

    Parameters
    ----------
    df_data : pd.DataFrame
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

    fig, axs = plt.subplots(2,2, sharex=True)

    cmap = plt.cm.coolwarm
    norm = CenteredNorm(vcenter=0, halfrange=abs(df_data["dc9"].min()))

    df_by_dc9 = df_data.groupby("dc9")

    for var, ax in zip(variables, axs):
        for dc9, df_group in df_by_dc9:
            plot_efficiency_curve(df_group, var, dc9, ax, cmap, norm, n, alpha)

    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', label=r'$\delta C_9$')

    out_dir_path = Path(out_dir_path)
    out_file_path = out_dir_path.joinpath("efficiency.png")
    plt.savefig(out_file_path, bbox_inches='tight')
    plt.close(fig)

    
