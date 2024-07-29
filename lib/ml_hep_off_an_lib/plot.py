
"""
Plotting functionality.
"""


import pathlib as pl
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


def setup_mpl_params():
    """
    Setup plotting parameters.
    
    i.e. Setup to make fancy looking plots.
    Inspiration from Chris Ketter.
    """
    mpl.rcParams["figure.figsize"] = (6, 4)
    mpl.rcParams["figure.dpi"] = 400
    mpl.rcParams["axes.titlesize"] = 11
    mpl.rcParams["figure.titlesize"] = 12
    mpl.rcParams["axes.labelsize"] = 14
    mpl.rcParams["figure.labelsize"] = 30
    mpl.rcParams["xtick.labelsize"] = 12 
    mpl.rcParams["ytick.labelsize"] = 12
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Computer Modern"]
    mpl.rcParams["axes.titley"] = None
    mpl.rcParams["axes.titlepad"] = 4
    mpl.rcParams["legend.fancybox"] = False
    mpl.rcParams["legend.framealpha"] = 0
    mpl.rcParams["legend.markerscale"] = 1
    mpl.rcParams["font.size"] = 7.5


def save_plot(path):
    """
    Save plot.
    """
    plt.savefig(path, bbox_inches="tight", transparent=True)


def setup_ax(ax, title=None, xlabel=None, ylabel=None, legend=True, xlim=(None,None), ylim=(None,None), yscale='linear'):
    """
    Setup the matplotlib axes object in terms of labels and scale.
    """
    
    ax.set_title(title, loc='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend: ax.legend()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yscale(yscale)


def ticks_in_radians(axis, interval:str):
    """
    Change axis ticks to multiples of pi.

    axis specifies the axis to change ('x' or 'y').
    interval specifies the interval of the ticks ('zero_to_two_pi' 
    or 'zero_to_pi')
    """

    zero_to_two_pi = {
        'nums':[0, np.pi/2, np.pi, (3/2)*np.pi, 2*np.pi],
        'syms':[r"$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    }         

    zero_to_pi = {
        'nums':[0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
        'syms': [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"]
    }

    if axis == "x":
        ticks = plt.xticks
    elif axis == "y":
        ticks = plt.yticks
        
    if interval == "0 to 2pi":
        ticks(zero_to_two_pi.nums, zero_to_two_pi.syms)
    elif interval == "0 to pi":
        ticks(zero_to_pi.nums, zero_to_pi.syms) 
    else: raise ValueError(f"Unrecognized interval: {interval}")


def stats_legend(
    dat_ser,
    descrp="",
    show_mean=True,
    show_count=True,
    show_rms=True,
):
    """
    Make a legend label similar to the roots stats box.

    Return a string meant to be used as a label for a matplotlib plot.
    Displayable stats are mean, count, and RMS.
    """
    
    def calculate_stats(ar):
        mean = np.mean(ar)
        count = count_events(ar)
        rms = np.std(ar)
        stats = {
            "mean": mean,
            "count": count,
            "rms": rms,
        }
        return stats
    
    stats = calculate_stats(dat_ser)

    leg = ""
    if descrp != "":
        leg += r"\textbf{" + f"{descrp}" + "}"
    if show_mean:
        leg += f"\nMean: {stats['mean']:.2G}"
    if show_count:
        leg += f"\nCount: {stats['count']}"
    if show_rms:
        leg += f"\nRMS: {stats['rms']:.2G}"

    return leg


def plot_hist(
    ax,
    data,
    desc,
    xlim=(None,None),
    density=True,
    scale='linear',
    histtype='step',
    color=None,
    stats=True,
    title=None,
    xlabel=None,
    save_path=None,
):
    """
    Plot a histogram.

    data can be a list of datasets.
    If data is a list of datasets, desc can be a list of descriptions
    and color can be a list of colors. 
    """

    if type(data) != list: data = [data]
    if type(desc) != list: desc = [desc]
    if type(color) != list: color = [color]
    assert len(data) == len(desc)

    bins = approx_bins(data, xlim=xlim)
    if color == [None]: color = [None]*len(data)
    if stats: label = [stats_legend(d, de) for d, de in zip(data, desc)]
    else: label = desc
    for d, l, c in zip(data, label, color):
        ax.hist(d, bins=bins, label=l, density=density, histtype=histtype, color=c)

    setup_ax(ax, title=title, xlabel=xlabel, yscale=yscale)
    if save_path: save_plot(save_path)
    


def plot_scatter(
    ax,
    d,
    desc,
    xlim=(None, None),
    ylim=(None, None),
    yscale='linear',
    color=None,
    info=None,
    title=None,
    xlabel=None,
    ylabel=None,
    save_path=None,
):    
    """
    Plot a scatter plot.

    data d is assumed to be a tuple of (xs, ys, errors) or a list of such tuples.
    If d is a list of tuples, desc must be a list of descriptions
    and color can be a list of colors. 
    info can be specified if additional legend info is to be shown (can be a list).
    """

    if type(d) != list: d = [d]
    if type(desc) != list: desc = [desc]
    if type(color) != list: color = [color]
    if type(info) != list: info = [info]
    if info == [None]: info = [None]*len(d)
    if color == [None]: color = [None]*len(d)
    assert len(d) == len(desc) == len(info)
    
    def label_desc(desc): return r"\textbf{" + desc + "}"
    def label_info(info): return "\n"+info
    label = [label_desc(de)+label_info(inf) if inf else label_desc(de) 
        for de, inf in zip(desc,info)]

    for dset, col, l in zip(d, color, label):
        ax.errorbar(dset[0], dset[1], yerr=dset[2], fmt='none', ecolor=col, elinewidth=0.5, capsize=1, alpha=0.7)
        ax.scatter(dset[0], dset[1], s=4, color=col, alpha=0.7, label=l, marker="X")
        
    setup_ax(ax, title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, yscale=yscale)
    if save_path: save_plot(save_path)
    

def plot_image(
    path_image_pickle_file,
    name_column_costheta_mu_bin,
    name_column_costheta_k_bin,
    name_column_chi_bin,
    name_column_q_squared,
    num_events,
):
    """Plot one of the images (i.e. angular distributions)."""

    image = pd.read_pickle(path_image_pickle_file)
    with mpl.rc_context():
        mpl.rcParams["figure.figsize"] = (6, 5)
        mpl.rcParams["axes.titlesize"] = 28
        mpl.rcParams["axes.labelsize"] = 24

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        sc = ax.scatter(
            image[name_column_costheta_mu_bin],
            image[name_column_costheta_k_bin],
            image[name_column_chi_bin],
            c=image[name_column_q_squared],
            cmap="magma",
        )
        ax.set_title(r"Average $q^2$ per Angular Bin")
        ax.set_xlabel(r"$\cos\theta_\mu$ Bin", labelpad=10)
        ax.set_ylabel(r"$\cos\theta_K$ Bin", labelpad=10)
        ax.set_zlabel(r"$\chi$ Bin", labelpad=10)
        ax.tick_params(pad=1)
        ax.annotate(
            f"Num. Events: {num_events}",
            (190, 250),
            xycoords="axes points",
            fontsize="xx-large",
        )
        fig.colorbar(
            sc,
            ax=ax,
            pad=0.025,
            shrink=0.6,
            location="left",
        )


def plot_eff(
    data_gen,
    data_det,
    num_points,
    title,
    xlabel,
    ylim=(0, 0.5),
    **kwargs,
):
    """Plot the efficiency."""

    def make_legend(num_gen, num_det):
        """Make the plot's legend."""
        legend = r"\textbf{Efficiency}" + f"\nDetector (signal): {num_det}\nGenerator: {num_gen}"
        return legend


    eff, bin_mids, err = calc_eff(data_gen, data_det, num_points)

    fig, ax = plt.subplots()

    num_gen = count_events(data_gen)
    num_det = count_events(data_det)
    legend = make_legend(num_gen, num_det)

    ax.scatter(
        bin_mids, 
        eff,
        label=legend,
        color="red",
        alpha=1,
        **kwargs,
    )
    ax.errorbar(
        bin_mids,
        eff,
        yerr=err,
        fmt="none",
        capsize=5,
        color="black",
        alpha=0.5,
        **kwargs,
    )

    ax.legend()
    set_x_lims(ax, data_gen, data_det)
    ax.set_ymargin(0.25)
    ax.set_ylim(ylim)
    ax.set_ylabel(r"$\varepsilon$", rotation=0, labelpad=20)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    return fig, ax


def plot_generator_efficiency(
    data_gen,
    data_gen_cut,
    num_points,
    title,
    xlabel,
    ylim=(0, 1),
    **kwargs,
):
    """Attempt to recreate efficiency plot using cuts on generator level data."""
  
    def make_legend(num_gen, num_gen_cut):
        """Make the plot's legend."""
        legend = r"\textbf{Generator Efficiency}" + f"\nCut Generator: {num_gen_cut}\nGenerator: {num_gen}"
        return legend

    eff, bin_mids, err = calc_eff(data_gen, data_gen_cut, num_points)

    fig, ax = plt.subplots()

    num_gen = count_events(data_gen)
    num_gen_cut = count_events(data_gen_cut)
    legend = make_legend(num_gen, num_gen_cut)

    ax.scatter(
        bin_mids, 
        eff,
        label=legend,
        color="red",
        alpha=0.5,
        **kwargs,
    )
    ax.errorbar(
        bin_mids,
        eff,
        yerr=err,
        fmt="none",
        capsize=5,
        color="black",
        alpha=0.5,
        **kwargs,
    )

    ax.legend()
    set_x_lims(ax, data_gen, data_gen_cut)
    ax.set_ymargin(0.25)
    ax.set_ylim(*ylim)
    ax.set_ylabel(r"$\varepsilon_\text{Gen}$", rotation=0, labelpad=20)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    return fig, ax



def plot_resolution(
    data,
    variable,
    q_squared_split,
    title,
    xlabel, 
    out_dir_path,
    xlim=(None, None),
    extra_name=""
):
    """Plot the resolution."""

    resolution = calculate_resolution(
        data,
        variable,
        q_squared_split,
    )

    if xlim not in {(None, None), None}:
        resolution = resolution[(resolution > xlim[0]) & (resolution < xlim[1])]

    fig, ax = plt.subplots()
    
    ax.hist(
        resolution,
        label=stats_legend(resolution, "Signal Events"),
        bins=approx_num_bins(resolution),
        color="red",
        histtype="step",
    )

    ax.legend()
    ax.set_xlim(xlim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)

    save_plot(
        plot_name=f'res_{variable}_{extra_name}',
        q_squared_split=q_squared_split,
        out_dir=out_dir_path
    )


def plot_candidate_multiplicity(data, out_dir_path, ell):
    """Plot a histogram of the candidate multiplicity."""

    sig = sig_(data).loc["det"]
    noise = noise_(data).loc["det"]

    plt.hist(
        sig["__event__"].value_counts().values, 
        bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], 
        label=stats_legend(sig["__event__"], descrp="Signal (after cuts)", show_mean=False, show_rms=False),
        color="red", 
        alpha=0.6,
        histtype="stepfilled"
    )
    
    plt.hist(
        noise["__event__"].value_counts().values, 
        bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], 
        label=stats_legend(noise["__event__"], descrp="Misrecon. (after cuts)", show_mean=False, show_rms=False),
        color="blue", 
        histtype="step"
    )
    if ell == "mu":
        plt.title(r"Candidate Multiplicity ($\ell = \mu$)")
    elif ell == "e":
        plt.title(r"Candidate Multiplicity ($\ell = e$)")
    else: raise Exception()
    plt.xlabel("Candidates per Event")
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.legend()

    save_plot(
        plot_name="cand_mult",
        q_squared_split="all",
        out_dir=out_dir_path
    )


def plot_hist_2d_side_by_side(
    xs,
    ys,
    num_bins,
    cmap
):
    """Plot two 2d-histograms side-by-side."""

    def subplots_side_by_side():
        fig, axs = plt.subplots(ncols=2, layout="constrained", sharey=True)
        return fig, axs

    def multi_hist_norm(hists):
        _, max_counts = min_max_over_arrays(hists)
        norm = plt.Normalize(0, max_counts)    
        return norm

    def calc_multi_hist_2d(xs: list, ys: list, num_bins: int):
        assert len(xs) == len(ys)
        xmin, xmax = min_max_over_arrays(xs)
        ymin, ymax = min_max_over_arrays(ys)
        interval = ((xmin, xmax), (ymin, ymax))
        hists = [np.histogram2d(x, y, num_bins, interval) for x, y in zip(xs, ys)]
        return hists
        
    def add_hist_2d(ax, hist_2d, norm, cmap):
        h = hist_2d[0]
        x_edges = hist_2d[1]
        y_edges = hist_2d[2]
        ax.pcolormesh(x_edges, y_edges, h.T, norm=norm, cmap=cmap)

    assert((len(xs) == 2) & (len(ys) == 2))

    hists = calc_multi_hist_2d(xs, ys, num_bins)
    hs = [hist[0] for hist in hists]
    norm = multi_hist_norm(hs)

    fig, axs = subplots_side_by_side()
    for hist, ax in zip(hists, axs):
        add_hist_2d(ax, hist, norm, cmap)

    return fig, axs, norm


def add_color_bar(axs, norm, cmap):
    """Add a colorbar scaled to multiple axes."""

    def scalar_mappable(norm, cmap):
        return mpl.cm.ScalarMappable(norm, cmap=cmap)
    
    sm = scalar_mappable(norm, cmap)
    plt.colorbar(sm, ax=axs, aspect=30)


def plot_volume_slices(ax_3d, arr, cmap, norm, n_slices=3):
    """
    Plot slices of volumetric data.
    Slices are along the z-axis (axis 2).
    Array arr should be a three-dimensional numpy array.
    Slices might not be evenly spaced along z-axis.
    """
    colors = cmap(norm(arr))
    
    x_axis = 0
    y_axis = 1
    z_axis = 2

    x_shape = arr.shape[x_axis]
    y_shape = arr.shape[y_axis]
    z_shape = arr.shape[z_axis]

    def xy_plane(z_pos):
        x, y = np.indices(
            (x_shape + 1, y_shape + 1)
        )
        z = np.full(
            (x_shape + 1, y_shape + 1), z_pos
        )
        return x, y, z

    def plot_slice(z_index):
        x, y, z = xy_plane(z_index) 

        ax_3d.plot_surface(
            x, y, z, 
            rstride=1, cstride=1, 
            facecolors=colors[:,:,z_index], 
            shade=False
        )

    def plot_outline(z_index, offset=0.3):
        x, y, z = xy_plane(z_index - offset)
        
        ax_3d.plot_surface(
            x, y, z, 
            rstride=1, cstride=1, 
            shade=False,
            color="#f2f2f2",
            edgecolor="#f2f2f2", 
        )

    z_indices = np.linspace(0, z_shape-1, n_slices, dtype=int)
    for i in z_indices:
        plot_outline(i)
        plot_slice(i)

