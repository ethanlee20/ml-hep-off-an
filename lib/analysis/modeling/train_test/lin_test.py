
"""
Functions for evaluating a model using the linearity test.
"""

from pathlib import Path
from math import sqrt
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader



def stats(x, y):
    """
    Calculate y's statistics for each unique x value.

    Parameters
    ----------
    x : array
    y : array
        Associated by order with x.

    Returns
    -------
    x_unique : list of numbers
        The unique x values.
    y_mean : list of numbers
        The mean of the ys for each unique x.
        Associated by order with x_unique.
    y_stdev : list of numbers
        The standard deviation for each unique x.
        Associated by order with x_unique.

    """

    assert len(x) == len(y)
    
    data = list(zip(x, y))
    
    x_unique = sorted(set(x))
    
    y_by_x = [[d[1] for d in data if d[0]==u] for u in x_unique]

    def stdev(s:list):
        m = mean(s)
        squares = map(lambda i: (i-m)**2, s)
        result = sqrt(sum(squares) / (len(s)-1))
        return result
    
    def sdm(s:list):
        std = stdev(s)
        result = std / sqrt(len(s))
        return result
    
    try:
        y_stderr = list(map(sdm, y_by_x))
    except ZeroDivisionError:
        y_stderr = 0

    y_mean = list(map(mean, y_by_x))

    return x_unique, y_mean, y_stderr


def plot_ref_line(ax, start_x, stop_x, buffer=0.05):
    """
    Parameters
    ----------
    ax : mpl.axes.Axes
        The axes on which to plot the reference line.
    start_x : float
        The x position of the start of the line.
    stop_x : float
        The x position of the stop of the line.
    buffer : float
        The x-axis distance to enlongate the line beyond the
        start_x and stop_x values. 

    Side Effects
    ------------
    - Plot a reference line on given axes.
    """
    ticks = np.linspace(start_x-buffer, stop_x+buffer, 2)
    ax.plot(
        ticks, ticks,
        label="Ref. Line (Slope = 1)",
        color="grey",
        linewidth=0.5,
        zorder=0
    )   


def plot_linearity(truth, pred_avg, pred_stderr, output_dirpath, run_name):
    """
    Plot the linearity test results.

    Parameters
    ----------
    truth : array of numbers
        The truth values.
    pred_avg : array of numbers
        The average prediction for each truth value.
        Associated by order with the truth values. 
    pred_stdev : array of numbers
        The standard deviation of the predictions for each truth value.
        Associated by order with the truth values.
    output_dirpath : str
        The path of the directory to which the plot is saved.
    run_name : str
        The identifier for the saved file.

    Side Effects
    ------------
    - Saves a plot.
    """
    fig, ax = plt.subplots()

    plot_ref_line(ax, start_x=truth[0], stop_x=truth[-1])

    ax.scatter(truth, pred_avg, label="Results on Val. Set", color="firebrick", zorder=5, s=16)
    ax.errorbar(truth, pred_avg, yerr=pred_stderr, fmt="none", elinewidth=0.5, capsize=0.5, color="black", label="Std. Err.", zorder=10)

    ax.set_xlim(-2.25, 1.35)
    ax.set_ylim(-2.25, 1.35)

    ax.legend()
    ax.set_xlabel(r"Actual $\delta C_9$")
    ax.set_ylabel(r"Predicted $\delta C_9$")
    ax.set_title(
        r"\textbf{Linearity} : " + f"{run_name}",
        loc="left"
    )

    # ax.text(
    #     1, 1.005, 
    #     r"\textbf{Events / Dist.} : " + f"{events_per_dist:.0e}\n" +
    #     r"\textbf{Test Dist. /} $\bm{\delta C_9}$ : " + f"{len(test_data)/len(list_dc9())}", 
    #     verticalalignment='bottom', 
    #     horizontalalignment='right',
    #     transform=ax.transAxes,
    # )

    output_dirpath = Path(output_dirpath)
    plot_filepath = output_dirpath.joinpath(f"{run_name}_lin.png")
    plt.savefig(plot_filepath, bbox_inches="tight")
    plt.close()    


def lin_test(model, test_data, device, output_dirpath, run_name):
    """
    Conduct a linearity test and plot the results.

    Parameters
    ----------
    model : torch.nn.Module
        The model to test.
    test_data : torch.utils.data.Dataset
        Testing dataset.
    device : str
        The device to train on.
        ("cuda" or "cpu")
    ouput_dirpath : str
        The path of the directory to which the
        linearity plot is saved.
    run_name : str
        The identifier for saved objects.

    Side Effects
    ------------
    - Plots the result of the linearity test.
    """

    test_dataloader = DataLoader(test_data, batch_size=1,)

    model = model.to(device)
    model.eval()
        
    def test(X, y):
        with torch.no_grad():
            pred = model(X.to(device))
            return y.item(), pred.item()
    outcomes = [test(X, y) for X, y in test_dataloader]
    truth, pred_avg, pred_stdev = stats(*list(zip(*outcomes)))

    plot_linearity(truth, pred_avg, pred_stdev, output_dirpath, run_name)




