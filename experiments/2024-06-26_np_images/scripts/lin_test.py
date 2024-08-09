
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from models import Convolutional3D, NN_Afb_S5, NeuralNetAvg

from helpers import ImageDataset, select_device, stats

from datasets import AvgsDataset

from ml_hep_off_an_lib.plot import setup_mpl_params
setup_mpl_params()


model = NeuralNetAvg()
level = "gen"
test_data = AvgsDataset("gen", train=False, events_per_dist=10_000)
test_dataloader = DataLoader(test_data, batch_size=1,)

device = select_device()

model = model.to(device)
model_state_dirpath = Path(f"../models/")
model_state_filepath = model_state_dirpath.joinpath(f"{model.name}.pt")
model.load_state_dict(torch.load(model_state_filepath))
model.eval()

        
def test(X, y):
    with torch.no_grad():
        pred = model(X.to(device))
        return y.item(), pred.item()
outcomes = [test(X, y) for X, y in test_dataloader]
truth, pred_avg, pred_stdev = stats(*list(zip(*outcomes)))

fig, ax = plt.subplots()

def plot_ref_line():
    buffer = 0.05
    ticks = np.linspace(truth[0]-buffer, truth[-1]+buffer, 5)
    plt.plot(
        ticks, ticks,
        label="Ref. Line (Slope = 1)",
        color="grey",
        linewidth=0.5,
        zorder=0
    )   
plot_ref_line()

ax.scatter(truth, pred_avg, label="Results on Val. Set", color="firebrick", zorder=5, s=16)
ax.errorbar(truth, pred_avg, yerr=pred_stdev, fmt="none", elinewidth=0.5, capsize=0.5, color="black", label="Std. Dev.", zorder=10)

ax.set_xlim(-2.25, 1.35)
ax.set_ylim(-2.25, 1.35)

ax.legend()
ax.set_xlabel(r"Actual $\delta C_9$")
ax.set_ylabel(r"Predicted $\delta C_9$")
ax.set_title(f"Linearity: {model.name}", loc="right")

plt.savefig(f"../plots/lin_{model.name}.png", bbox_inches="tight")
plt.close()