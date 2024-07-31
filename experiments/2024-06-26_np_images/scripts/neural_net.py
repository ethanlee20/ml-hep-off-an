
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

from ml_hep_off_an_lib.plot import setup_mpl_params

from helpers import stats, ImageDataset, test_loop, train_loop


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10*10*10, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
        )
        self.double()
    
    def forward(self, x):
        x = self.flatten(x)
        result = self.linear_relu_stack(x)
        return result
    

def main():
    for level in ["gen", "det"]:
        learning_rate = 1e-3
        epochs = 100
        train_batch_size = 20
        test_batch_size = 5
        
        device = (
            "cuda" 
            if torch.cuda.is_available()
            else 
            "cpu"
        )

        model = NeuralNetwork().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_data = ImageDataset(level, train=True,)
        test_data = ImageDataset(level, train=False,)

        train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

        avg_test_losses = []
        avg_train_losses = []
        for t in range(epochs):
            # if t == 50:
                # breakpoint()
            avg_train_losses.append(train_loop(train_dataloader, model, loss_fn, optimizer, device))
            avg_test_losses.append(test_loop(test_dataloader, model, loss_fn, device))

        setup_mpl_params()

        def plot_losses(test_losses, train_losses):
            plt.plot(test_losses, label="test")
            plt.plot(train_losses, label="train")
            plt.yscale("log")
            plt.xlabel("Epoch")
            plt.ylabel(r"Loss (MSE)")
            plt.title(f"Test batch size: {test_batch_size}\nTrain batch size: {train_batch_size}", loc="right")
            plt.legend()
            plt.savefig(f"../plots/nn_losses_{level}.png", bbox_inches="tight")
            plt.close()

        plot_losses(avg_test_losses, avg_train_losses)

        def plot_lin(model, dataloader):
            ys = []
            predicts = []
            for X, y in dataloader:
                p = model(X.to(device)).squeeze(1).tolist()
                predicts += p
                y = y.tolist()
                ys += y
            
            y_ticks, pred_avg, pred_stdev = stats(ys, predicts)
            
            def plot_ref_line():
                buffer = 0.05
                ticks = np.linspace(y_ticks[0]-buffer, y_ticks[-1]+buffer, 5)
                plt.plot(
                    ticks, ticks,
                    label="Ref. Line (Slope = 1)",
                    color="grey",
                    linewidth=0.5,
                    zorder=0
                )
            
            plot_ref_line()
            
            plt.scatter(y_ticks, pred_avg, label="Results on Val. Set", color="firebrick", zorder=5, s=16)
            plt.errorbar(y_ticks, pred_avg, yerr=pred_stdev, fmt="none", elinewidth=0.5, capsize=0.5, color="black", label="Std. Dev.", zorder=10)
            
            plt.xlim(-2.25, 1.35)
            plt.ylim(-2.25, 1.35)

            plt.legend()
            plt.xlabel(r"Actual $\delta C_9$")
            plt.ylabel(r"Predicted $\delta C_9$")
            
            def make_title(level):
                result = r"$\mu$"
                if level == "gen":
                    result += ", Generator" 
                elif level=="det":
                    result += ", Detector"  
                result += f", Fully Connected\nEpochs: {epochs}, Learn. Rate: {learning_rate}"
                result += f"\nBatch Size: {train_batch_size} (Train), {test_batch_size} (Test)"
                return result
            
            plt.title(make_title(level), loc="right")
            
            plt.savefig(f"../plots/nn_lin_{level}.png", bbox_inches="tight")
            plt.close()
            
        plot_lin(model, test_dataloader)


if __name__ == "__main__":
    main()