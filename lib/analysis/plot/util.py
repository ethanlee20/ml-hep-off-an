
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

    Side Effects
    ------------
    - Changes matplotlib rc parameters.
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
    mpl.rcParams["text.latex.preamble"] = r"\usepackage{bm}"
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Computer Modern"]
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.titley"] = None
    mpl.rcParams["axes.titlepad"] = 4
    mpl.rcParams["legend.fancybox"] = False
    mpl.rcParams["legend.framealpha"] = 0
    mpl.rcParams["legend.markerscale"] = 1
    mpl.rcParams["legend.fontsize"] = 7.5


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