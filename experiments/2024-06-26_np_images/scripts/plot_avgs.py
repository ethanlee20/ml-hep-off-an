

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from ml_hep_off_an_lib.plot import setup_mpl_params
setup_mpl_params()

from datasets import AvgsDataset

from helpers import list_dc9

events_per_dist = 40_000
sampling_ratio = None
level="det"

dset = AvgsDataset(level, train=True, events_per_dist=events_per_dist, sampling_ratio=sampling_ratio, std_scale=False)

num_dists_per_dc9 = len(dset)/len(list_dc9()) 

dloader = DataLoader(dset, batch_size=1)

dc9s = []
q_squareds = []
costheta_mus = []
costheta_ks = []
chis = []

for X, y in dloader:
    # breakpoint()
    dc9s.append(y.item())
    q_squareds.append(X[:,0].item())
    costheta_mus.append(X[:,1].item())
    costheta_ks.append(X[:,2].item())
    chis.append(X[:,3].item())
    
fig, axs = plt.subplots(2,2, sharex=True)

alpha = 0.7
size = 4

axs[0,0].scatter(dc9s, q_squareds, alpha=alpha, s=size)
axs[0,0].set_ylabel(r"${q^2}_{avg}$ [GeV$^2$]")
axs[0,0].set_ylim((8.76, 11))
axs[0,1].scatter(dc9s, costheta_mus, alpha=alpha, s=size)
axs[0,1].set_ylabel(r"${\cos\theta_\mu}_{avg}$")
axs[0,1].set_ylim((0.08, 0.21))
axs[1,0].scatter(dc9s, costheta_ks, alpha=alpha, s=size)
axs[1,0].set_ylabel(r"${\cos\theta_K}_{avg}$")
axs[1,0].set_xlabel(r"$\delta C_9$")
axs[1,0].set_ylim((-0.05, 0.02))
axs[1,1].scatter(dc9s, chis, alpha=alpha, s=size)
axs[1,1].set_ylabel(r"${\chi}_{avg}$")
axs[1,1].set_xlabel(r"$\delta C_9$")
axs[1,1].set_ylim((3.05, 3.23))

axs[0,1].text(
    1, 1.005, 
    r"\textbf{Events / Dist.} : " + f"{events_per_dist:.0e}\n" +
    r"\textbf{Dist. /} $\bm{\delta C_9}$ : " + f"{num_dists_per_dc9:.2n}", 
    verticalalignment='bottom', 
    horizontalalignment='right',
    transform=axs[0,1].transAxes,
)
axs[0,0].text(
    0, 1.005, 
    r"\textbf{" + ("Generator" if level=="gen" else "Detector") + "}",
    verticalalignment='bottom', 
    horizontalalignment='left',
    transform=axs[0,0].transAxes,
)

fig.tight_layout()

plt.savefig(f"../plots/avgs_{level}.png", bbox_inches="tight")
plt.close()

# breakpoint()