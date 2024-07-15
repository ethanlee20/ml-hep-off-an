
from math import pi

import pathlib as pl
import matplotlib as mpl
import matplotlib.pyplot as plt

from mylib.phys import calc_eff
from mylib.util import section, open_data
from mylib.plot import setup_mpl_params, plot_scatter, save_plot

# plt.style.use('./dark_style.mplstyle')
setup_mpl_params()

for ell in {'mu', 'e'}:

    path_data = pl.Path(f"/home/belle2/elee20/ml-hep-proj/experiments/2024-06-04_eff_res/datafiles/{ell}/an")
    path_output_dir = pl.Path("/home/belle2/elee20/ml-hep-proj/experiments/2024-06-04_eff_res/plots")
    path_output_dir.mkdir(parents=True, exist_ok=True)

    data = open_data(path_data)

    data_gen = section(data, sig_noise='sig', gen_det='gen')
    data_det = section(data, sig_noise='sig', gen_det='det')

    fig, axs = plt.subplots(3, 1)

    legend_info = f"Num. Gen: {len(data_gen)}\nNum. Det: {len(data_det)}"
    if ell == 'mu': ell_sym = r'\mu'
    else: ell_sym = r'e'

    variables = [
        "chi",
        f"costheta_{ell}",
        "costheta_K",
    ]

    xlim = [
        (0, 2*pi),
        (-1, 1),
        (-1, 1)
    ]

    xlabel = [
        r"$\chi$",
        r"$\cos\theta_"+ell_sym+"$",
        r"$\cos\theta_K$",
    ]

    for v, xli, xla, ax in zip(variables, xlim, xlabel, axs):
        eff = calc_eff(data_gen[v], data_det[v], n=50)
        plot_scatter(
            ax, 
            eff, 
            "Efficiency", 
            xlim=xli,
            ylim=(0, 0.5),
            title=f"Efficiency, "+"$"+ell_sym+"$",
            xlabel=xla,
            ylabel=r"$\epsilon$",
            info=legend_info,
            color='powderblue',
        )
    save_plot(path_output_dir.joinpath(f"{ell}_efficiency.png"))
    plt.close()


