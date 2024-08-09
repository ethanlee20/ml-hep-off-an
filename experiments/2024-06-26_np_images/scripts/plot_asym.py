
from matplotlib.colors import CenteredNorm
from matplotlib.cm import ScalarMappable

import matplotlib.pyplot as plt

from ml_hep_off_an_lib.afb import calc_afb_of_q_squared
from ml_hep_off_an_lib.s5 import calc_s5_of_q_squared
from ml_hep_off_an_lib.plot import setup_mpl_params

from helpers import load_df_all_trials, list_dc9


def plot_afb(data, dc9, ax, cmap, norm, ell='mu', num_points=3, alpha=0.8):
    x, y, err = calc_afb_of_q_squared(data, ell, num_points)
    label = r"$\delta C_9 = " + f"{dc9}$"
    color = cmap(norm(dc9), alpha=alpha)
    ax.errorbar(x, y, yerr=err, fmt='none', ecolor=color, elinewidth=0.5, capsize=0)
    ax.scatter(x, y, s=4, color=color, label=label)
    return


def plot_s5(data, dc9, ax, cmap, norm, ell='mu', num_points=3, alpha=0.8):
    x, y, err = calc_s5_of_q_squared(data, num_points)
    label = r"$\delta C_9 = " + f"{dc9}$"
    color = cmap(norm(dc9), alpha=alpha)
    ax.errorbar(x, y, yerr=err, fmt='none', ecolor=color, elinewidth=0.5, capsize=0)
    ax.scatter(x, y, s=4, color=color, label=label)
    return


def main():

    setup_mpl_params()

    dc9_values = list_dc9()

    cmap = plt.cm.coolwarm
    norm = CenteredNorm(vcenter=0, halfrange=abs(dc9_values[0]))

    levels = ['gen', 'det']
    titles = [r"$\mu$, Generator, $360$k / $\delta C_9$", r"$\mu$, Detector (sig. only), $\sim 120$k / $\delta C_9$"]

    for level, title in zip(levels, titles):
        
        # AFB
        fig, ax = plt.subplots()
        
        for dc9 in dc9_values:
            data = load_df_all_trials(dc9).loc[level]
            plot_afb(data, dc9, ax, cmap, norm)

        ax.set_title(title, loc='right')
        ax.set_ylabel(r'$A_{FB}$')
        ax.set_xlabel(r'$q^2$ [GeV$^2$]')
        ax.set_xbound(0, 19)
        ax.set_ybound(-0.3, 0.5)
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', label=r'$\delta C_9$')
        
        plt.savefig(f"../plots/afb_{level}_lowbin.png", bbox_inches='tight')
        plt.close(fig)

        # S5
        fig, ax = plt.subplots()
        
        for dc9 in dc9_values:
            data = load_df_all_trials(dc9).loc[level]
            plot_s5(data, dc9, ax, cmap, norm)

        ax.set_title(title, loc='right')
        ax.set_ylabel(r'$S_{5}$')
        ax.set_xlabel(r'$q^2$ [GeV$^2$]')
        ax.set_xbound(0, 19)
        ax.set_ybound(-0.5, 0.5)
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', label=r'$\delta C_9$')
        
        plt.savefig(f"../plots/s5_{level}_lowbin.png", bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    main()