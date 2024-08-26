
"""
Training a model (and outputting the loss curves).
"""
from math import sqrt
from statistics import mean
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from helpers import ImageDataset, train_loop, test_loop, select_device
from models import Convolutional3D, NN_Afb_S5, NeuralNetAvg, Conv2dNet
from datasets import AvgsDataset, Hist2dDataset

from analysis.plot import setup_mpl_params
setup_mpl_params()

from analysis.models.deep_sets import Deep_Sets
from analysis.models.pp import NN_pp
from analysis.models.te import TransformerRegressor
from analysis.datasets.as_is import As_Is_Dataset
from analysis.datasets.per_part import Per_Part_Dataset





def train(model, train_data:Dataset, test_data:Dataset, output_dirpath, learning_rate=1e-3, epochs=100, train_batch_size=20, test_batch_size=5):
    
    output_dirpath = Path(output_dirpath)

    device = select_device()
    model = model.to(device)
        
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

    loss = []   
    for t in range(epochs):
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loss = test_loop(test_dataloader, model, loss_fn, device)
        loss.append({"epoch": t, "train_loss": train_loss, "test_loss": test_loss})
        print(f"epoch: {t}\ntrain_loss: {train_loss}\ntest_loss: {test_loss}")
    df_loss = pd.DataFrame.from_records(loss).iloc[2:]

    output_model_filepath = output_dirpath.joinpath(f"{"pp"}.pt")
    torch.save(model.state_dict(), output_model_filepath)

    # Loss Plot
    fig, ax = plt.subplots()
    ax.plot(df_loss["epoch"], df_loss["train_loss"], label="Training Loss")
    ax.plot(df_loss["epoch"], df_loss["test_loss"], label="Validation Loss")
    ax.legend()
    ax.set_title(f"Final Test Loss: {df_loss["test_loss"].iloc[-1]}", loc="right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")

    output_loss_filepath = output_dirpath.joinpath(f"{"pp"}_loss.png")
    plt.savefig(output_loss_filepath, bbox_inches="tight")
    plt.close()




    def stats(x:list, y:list):
        """
        Calculate statistics of y values for each x value.
        Return [x_ticks], [y_mean], [y_stdev]
        """
        
        assert len(x) == len(y)
        
        data = list(zip(x, y))
        
        x_ticks = sorted(set(x))
        
        y_by_x = [[d[1] for d in data if d[0]==tick] for tick in x_ticks]

        def stdev(s:list):
            m = mean(s)
            sq = map(lambda i: (i-m)**2, s)
            result = sqrt(sum(sq) / (len(s)-1))
            return result
        
        y_stdev = list(map(stdev, y_by_x))

        y_mean = list(map(mean, y_by_x))

        return x_ticks, y_mean, y_stdev

    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

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
    ax.set_title(
        r"\textbf{Linearity} : " + f"{"pp"}, {""}",
        loc="left"
        )

    ax.text(
        1, 1.005, 
        r"\textbf{Events / Dist.} : " + f"{0:.0e}\n" +
        r"\textbf{Test Dist. /} $\bm{\delta C_9}$ : " + f"{len(test_data)/1}", 
        verticalalignment='bottom', 
        horizontalalignment='right',
        transform=ax.transAxes,
    )

    plt.savefig(f"../plots/lin_{"pp"}.png", bbox_inches="tight")
    plt.close()



    
    return




def main():
    
    # data_dirpath = "../datafiles/hist_4d_with_afb_s5"
    # level = "det"
    # train_data = ImageDataset(level=level, train=True, dirpath=data_dirpath)
    # test_data = ImageDataset(level=level, train=False, dirpath=data_dirpath)

    # model = Convolutional3D(f"cnn3d_with_afb_s5_{level}")

    # train(model, train_data, test_data, output_dirpath="../models", learning_rate=1e-4)

    level = "gen"
    train_data = Per_Part_Dataset(level, train=True)
    test_data = Per_Part_Dataset(level, train=False)
    print(train_data.data)

    model = NN_pp()

    # breakpoint()

    train(
        model, 
        train_data, 
        test_data, 
        output_dirpath="../models", 
        learning_rate=1e-3, 
        epochs=10, 
        train_batch_size=42, 
        test_batch_size=32,
    )

#     """
    
# (ml) C:\Users\tetha\Desktop\ml-hep-off-an\experiments\2024-06-26_np_images\scripts>python train.py
# [[ 0.27314843  0.82957253  0.21713193  1.17828471 -1.41      ]
#  [-0.97489499  0.56692187  0.72777079 -0.60024411 -0.67      ]
#  [-0.21752522 -1.70248751  0.66602261  0.226454   -0.38      ]
#  [ 1.57881464  0.00278105  0.11161887  0.56637671 -1.78      ]
#  [-0.65954287  0.30321205 -1.7225442  -1.37087131 -1.7       ]]
#     """


if __name__ == "__main__":
    main()