
from math import pi
import pathlib as pl
import matplotlib.pyplot as plt

from mylib.util import open_data, section, approx_bins
from mylib.plot import setup_mpl_params_save, save_plot, stats_legend


def plot_hist(
    ax,     
    data, 
    desc,
    xlim=(None,None),
    density=True, 
    histtype='step', 
    color=None,
    stats=True,
):
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



setup_mpl_params_save()

path_sm_data = pl.Path("/home/belle2/elee20/ml-hep-proj/experiments/2024-05-22_newphys/run1/datafiles/sm_an")
path_np_data = pl.Path("/home/belle2/elee20/ml-hep-proj/experiments/2024-05-22_newphys/run1/datafiles/np_an")
path_output_dir = pl.Path("/home/belle2/elee20/ml-hep-proj/experiments/2024-05-22_newphys/run1/plots")

path_output_dir.mkdir(parents=True, exist_ok=True)

sm_data = open_data(path_sm_data)
np_data = open_data(path_np_data)

q_squared_splits = [
    "all",
    "med",
]

titles = [
    r"($\mu$, NP: $\delta C_9 = -0.87$, All $q^2$)",
    r"($\mu$, NP: $\delta C_9 = -0.87$, $1 < q^2 < 6$ GeV$^2$)",
]

vars = [
    "deltaE",
    "Mbc",
    "q_squared",
    "costheta_mu",
    "costheta_K",
    "chi",
]

xlabels = [
    r"$\Delta E$ [GeV]",
    r"$M_{bc}$ [GeV$^2$]",
    r"$q^2$ [GeV$^2$]",
    r"$\cos\theta_\mu$",
    r"$\cos\theta_K$",
    r"$\chi$",
]

xlims = [
    (-0.05, 0.05),
    (5.27, 5.5),
    (0, 20),    
    (-1, 1),
    (-1, 1),
    (0, 2*pi),
]

for q_squared_split, title in zip(q_squared_splits, titles):

    for var, xlabel, xlim in zip(vars, xlabels, xlims):

        sm_data_gen = section(sm_data, var=var, q_squared_split=q_squared_split, sig_noise='sig', gen_det='gen')
        sm_data_det = section(sm_data, var=var, q_squared_split=q_squared_split, sig_noise='sig', gen_det='det')
        np_data_gen = section(np_data, var=var, q_squared_split=q_squared_split, sig_noise='sig', gen_det='gen')
        np_data_det = section(np_data, var=var, q_squared_split=q_squared_split, sig_noise='sig', gen_det='det')
    
        fig, ax = plt.subplots()

        plot_hist(
            ax,
            [sm_data_gen, np_data_gen], 
            ["Gen. SM", "Gen. NP"],
            xlim=xlim,
            color=["red", "blue"],
        )
        ax.legend()
        ax.set_title(title, loc='right')
        ax.set_xlabel(xlabel)
        save_plot(f"{var}_gen", q_squared_split, path_output_dir)

        fig, ax = plt.subplots()

        plot_hist(
            ax,
            [sm_data_det, np_data_det], 
            ["Det. (sig.) SM", "Det. (sig.) NP"],    
            xlim=xlim,
            color=["red", "blue"],
        )
        ax.legend()
        ax.set_title(title, loc='right')
        ax.set_xlabel(xlabel)
        save_plot(f"{var}_det", q_squared_split, path_output_dir)
        


