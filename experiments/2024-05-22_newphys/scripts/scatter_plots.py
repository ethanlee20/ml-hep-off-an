
import pathlib as pl
import matplotlib as mpl
import matplotlib.pyplot as plt

from mylib.phys import calc_afb_of_q_squared, calc_s5_of_q_squared
from mylib.util import section, open_data
from mylib.plot import save_plot, stats_legend, setup_mpl_params_save


num_points = 150
ell = "mu"

setup_mpl_params_save()
mpl.rcParams["legend.markerscale"] = 1
title = r"($\mu$, NP: $\delta C_9 = -0.87$)"

path_sm_data = pl.Path("/home/belle2/elee20/ml-hep-proj/experiments/2024-05-22_newphys/run1/datafiles/sm_an")
path_np_data = pl.Path("/home/belle2/elee20/ml-hep-proj/experiments/2024-05-22_newphys/run1/datafiles/np_an")
path_output_dir = pl.Path("/home/belle2/elee20/ml-hep-proj/experiments/2024-05-22_newphys/run1/plots")
path_output_dir.mkdir(parents=True, exist_ok=True)

sm_data = open_data(path_sm_data)
np_data = open_data(path_np_data)

sm_data_gen = section(sm_data, sig_noise='sig', gen_det='gen')
sm_data_det = section(sm_data, sig_noise='sig', gen_det='det')
np_data_gen = section(np_data, sig_noise='sig', gen_det='gen')
np_data_det = section(np_data, sig_noise='sig', gen_det='det')

leg_gen_np = stats_legend(np_data_gen, "Gen. NP", show_mean=False, show_rms=False)
leg_det_np = stats_legend(np_data_det, "Det. (sig.) NP", show_mean=False, show_rms=False)
leg_gen_sm = stats_legend(sm_data_gen, "Gen. SM", show_mean=False, show_rms=False)
leg_det_sm = stats_legend(sm_data_det, "Det. (sig.) SM", show_mean=False, show_rms=False)

afb_gen_np_x, afb_gen_np_y, afb_gen_np_err = calc_afb_of_q_squared(np_data_gen, ell, num_points)
afb_det_np_x, afb_det_np_y, afb_det_np_err = calc_afb_of_q_squared(np_data_det, ell, num_points)
afb_gen_sm_x, afb_gen_sm_y, afb_gen_sm_err = calc_afb_of_q_squared(sm_data_gen, ell, num_points)
afb_det_sm_x, afb_det_sm_y, afb_det_sm_err = calc_afb_of_q_squared(sm_data_det, ell, num_points)

plt.errorbar(afb_gen_sm_x, afb_gen_sm_y, yerr=afb_gen_sm_err, fmt='none', ecolor='red', elinewidth=0.5, capsize=1, alpha=0.7)
plt.scatter(afb_gen_sm_x, afb_gen_sm_y, s=4, color="red", alpha=0.7, label=leg_gen_sm, marker="X")
plt.errorbar(afb_gen_np_x, afb_gen_np_y, yerr=afb_gen_np_err, fmt='none', ecolor='blue', elinewidth=0.5, capsize=1, alpha=0.7)
plt.scatter(afb_gen_np_x, afb_gen_np_y, s=4, color="blue", alpha=0.7, label=leg_gen_np, marker="d")

plt.ylim(-0.4, 0.6)
plt.xlim(0, 19) 
plt.title(title, loc="right")
plt.xlabel(r"$q^2$ [GeV$^2$]")
plt.ylabel(r"$A_{FB}$")
plt.legend()
save_plot("afb_gen", "all", path_output_dir)

plt.errorbar(afb_det_sm_x, afb_det_sm_y, yerr=afb_det_sm_err, fmt='none', ecolor='red', elinewidth=0.5, capsize=1, alpha=0.7)
plt.scatter(afb_det_sm_x, afb_det_sm_y, s=4, color="red", alpha=0.7, label=leg_det_sm, marker="X")
plt.errorbar(afb_det_np_x, afb_det_np_y, yerr=afb_det_np_err, fmt='none', ecolor='blue', elinewidth=0.5, capsize=1, alpha=0.7)
plt.scatter(afb_det_np_x, afb_det_np_y, s=4, color="blue", alpha=0.7, label=leg_det_np, marker="d")

plt.ylim(-0.4, 0.6)
plt.xlim(0, 19) 
plt.title(title, loc="right")
plt.xlabel(r"$q^2$ [GeV$^2$]")
plt.ylabel(r"$A_{FB}$")
plt.legend()
save_plot("afb_det", "all", path_output_dir)

s5_gen_np_x, s5_gen_np_y, s5_gen_np_err = calc_s5_of_q_squared(np_data_gen, num_points)
s5_det_np_x, s5_det_np_y, s5_det_np_err = calc_s5_of_q_squared(np_data_det, num_points)
s5_gen_sm_x, s5_gen_sm_y, s5_gen_sm_err = calc_s5_of_q_squared(sm_data_gen, num_points)
s5_det_sm_x, s5_det_sm_y, s5_det_sm_err = calc_s5_of_q_squared(sm_data_det, num_points)

plt.errorbar(s5_gen_sm_x, s5_gen_sm_y, yerr=s5_gen_sm_err, fmt='none', ecolor='red', elinewidth=0.5, capsize=1, alpha=0.7)
plt.scatter(s5_gen_sm_x, s5_gen_sm_y, s=4, color="red", alpha=0.7, label=leg_gen_sm, marker="X")
plt.errorbar(s5_gen_np_x, s5_gen_np_y, yerr=s5_gen_np_err, fmt='none', ecolor='blue', elinewidth=0.5, capsize=1, alpha=0.7)
plt.scatter(s5_gen_np_x, s5_gen_np_y, s=4, color="blue", alpha=0.7, label=leg_gen_np, marker="d")

plt.ylim(-0.65, 0.45)
plt.xlim(0, 19) 
plt.title(title, loc="right")
plt.xlabel(r"$q^2$ [GeV$^2$]")
plt.ylabel(r"$S_5$")
plt.legend()
save_plot("s5_gen", "all", path_output_dir)

plt.errorbar(s5_det_sm_x, s5_det_sm_y, yerr=s5_det_sm_err, fmt='none', ecolor='red', elinewidth=0.5, capsize=1, alpha=0.7)
plt.scatter(s5_det_sm_x, s5_det_sm_y, s=4, color="red", alpha=0.7, label=leg_det_sm, marker="X")
plt.errorbar(s5_det_np_x, s5_det_np_y, yerr=s5_det_np_err, fmt='none', ecolor='blue', elinewidth=0.5, capsize=1, alpha=0.7)
plt.scatter(s5_det_np_x, s5_det_np_y, s=4, color="blue", alpha=0.7, label=leg_det_np, marker="d")

plt.ylim(-0.65, 0.45)
plt.xlim(0, 19) 
plt.title(title, loc="right")
plt.xlabel(r"$q^2$ [GeV$^2$]")
plt.ylabel(r"$S_5$")
plt.legend()
save_plot("s5_det", "all", path_output_dir)
