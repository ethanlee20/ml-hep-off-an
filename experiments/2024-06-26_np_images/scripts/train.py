
"""
Training a model (and outputting the loss curves).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from helpers import ImageDataset, train_loop, test_loop, select_device
from models import Convolutional3D, NN_Afb_S5, NeuralNetAvg
from datasets import AvgsDataset

from ml_hep_off_an_lib.plot import setup_mpl_params
setup_mpl_params()





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
        # if t % 5 == 0:
        print(f"epoch: {t}\ntrain_loss: {train_loss}\ntest_loss: {test_loss}")
    df_loss = pd.DataFrame.from_records(loss)

    output_model_filepath = output_dirpath.joinpath(f"{model.name}.pt")
    torch.save(model.state_dict(), output_model_filepath)

    # Loss Plot
    fig, ax = plt.subplots()
    ax.plot(df_loss["epoch"], df_loss["train_loss"], label="Training Loss")
    ax.plot(df_loss["epoch"], df_loss["test_loss"], label="Validation Loss")
    ax.legend()
    ax.set_title(f"Final Test Loss: {df_loss["test_loss"].iloc[-1]}", loc="right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")

    output_loss_filepath = output_dirpath.joinpath(f"{model.name}_loss.png")
    plt.savefig(output_loss_filepath)
    plt.close()
    
    return




def main():
    
    # data_dirpath = "../datafiles/hist_4d_with_afb_s5"
    # level = "det"
    # train_data = ImageDataset(level=level, train=True, dirpath=data_dirpath)
    # test_data = ImageDataset(level=level, train=False, dirpath=data_dirpath)

    # model = Convolutional3D(f"cnn3d_with_afb_s5_{level}")

    # train(model, train_data, test_data, output_dirpath="../models", learning_rate=1e-4)


    train_data = AvgsDataset("gen", train=True, events_per_dist=20_000, sampling_ratio=1.5)
    test_data = AvgsDataset("gen", train=False, events_per_dist=20_000)

    model = NeuralNetAvg()
    train(
        model, 
        train_data, 
        test_data, 
        output_dirpath="../models", 
        learning_rate=1e-3, 
        epochs=100, 
        train_batch_size=32, 
        test_batch_size=16,
    )

    


if __name__ == "__main__":
    main()