
"""
Functions for model training and evaluation.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader



def train_loop(dataloader, model, loss_fn, optimizer, device):
    """
    Train a model using gradient descent.

    Train for an epoch.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Train data dataloader.
    model : torch.nn.Module
        The model to train.
    loss_fn : torch.nn.modules.loss._Loss
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimizer.
    device : str
        The device to train on.
        ("cuda" or "cpu")
        
    Returns
    -------
    float
        Average batch training loss over the epoch.
    
    Side Effects
    ------------
    Updates model parameters (in-place) using gradient descent.
    """

    def train(X, y):
        model.train()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    
    num_batches = len(dataloader)
    losses = [train(X, y) for X, y in dataloader]
    avg_train_loss = sum(losses) / num_batches
    return avg_train_loss


def test_loop(dataloader, model, loss_fn, device):
    """
    Evaluate a model's performance on test data.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Test data dataloader.
    model : torch.nn.Module
        The model to test.
    loss_fn : torch.nn.modules.loss._Loss
        The loss function.
    device : str
        The device to train on.
        ("cuda" or "cpu")
        
    Returns
    -------
    float
        Average batch training loss over the test dataset.
    """
    def test(X, y):
        model.eval()
        with torch.no_grad():
            pred = model(X)
            loss = loss_fn(pred, y)
            return loss.item()
    
    num_batches = len(dataloader)
    losses = [test(X, y) for X, y in dataloader]
    avg_test_loss = sum(losses) / num_batches
    
    return avg_test_loss


def train_test(model, train_data, test_data, loss_fn, optimizer, device, output_dirpath, run_name, epochs=100, train_batch_size=20, test_batch_size=5):
    """
    Train a model using gradient descent.

    Evaluate the model's performance on test data after every epoch.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    train_data : torch.utils.data.Dataset
        Training dataset.
    test_data : torch.utils.data.Dataset
        Testing dataset.
    loss_fn : torch.nn.modules.loss._Loss
        The loss function.
    optimizer : torch.optim.Optimizer
        The optimizer.
    device : str
        The device to train on.
        ("cuda" or "cpu")
    output_dirpath : str
        The path of the directory to which the
        trained model and loss plot are saved.
    run_name : str
        The identifier for saved objects.
    epochs : int, optional
        The number of epochs to train for
        (default 100).
    train_batch_size : int, optional
        The number of examples per training batch
        (default 20).
    test_batch_size : int, optional
        The number of examples per test batch
        (default 5).

    Side Effects
    ------------
    - Updates model parameters (in-place) using gradient descent.
    - Saves a plot of the loss.
    - Saves the model parameters.
    """
    output_dirpath = Path(output_dirpath)

    model = model.to(device)
        
    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

    loss = []   
    for t in range(epochs):
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loss = test_loop(test_dataloader, model, loss_fn, device)
        loss.append({"epoch": t, "train_loss": train_loss, "test_loss": test_loss})
        print(f"epoch: {t}\ntrain_loss: {train_loss}\ntest_loss: {test_loss}")
    df_loss = pd.DataFrame.from_records(loss).iloc[2:]

    output_model_filepath = output_dirpath.joinpath(f"{run_name}.pt")
    torch.save(model.state_dict(), output_model_filepath)

    # Loss Plot
    fig, ax = plt.subplots()
    ax.plot(df_loss["epoch"], df_loss["train_loss"], label="Training Loss")
    ax.plot(df_loss["epoch"], df_loss["test_loss"], label="Validation Loss")
    ax.legend()
    ax.set_title(f"Final Test Loss: {df_loss["test_loss"].iloc[-1]}", loc="right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")

    loss_plot_filepath = output_dirpath.joinpath(f"{run_name}_loss.png")
    plt.savefig(loss_plot_filepath, bbox_inches="tight")
    plt.close()
