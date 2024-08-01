"""
Training a model.
"""

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

from ml_hep_off_an_lib.plot import setup_mpl_params

from helpers import ImageDataset, train_loop, test_loop, stats
from cnn import ConvolutionalNN


def select_device():
    device = (
        "cuda" 
        if torch.cuda.is_available()
        else 
        "cpu"
    )
    print("Device: ", device)
    return device






def main():
      
    model = ConvolutionalNN()
    learning_rate = 1e-3
    epochs = 100
    train_batch_size = 20
    test_batch_size = 5
    level = "gen"
    
    device = select_device()
    model = model.to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_data = ImageDataset(level, train=True,)
    test_data = ImageDataset(level, train=False,)

    train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)
    
    for t in range(epochs):
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loss = test_loop(test_dataloader, model, loss_fn, device)

    

        # setup_mpl_params()

        # def plot_losses(test_losses, train_losses):
        #     plt.plot(test_losses, label="test")
        #     plt.plot(train_losses, label="train")
        #     plt.yscale("log")
        #     plt.xlabel("Epoch")
        #     plt.ylabel(r"Loss (MSE)")
        #     plt.title(f"Test batch size: {test_batch_size}\nTrain batch size: {train_batch_size}", loc="right")
        #     plt.legend()
        #     plt.savefig(f"../plots/cnn_losses_{level}.png", bbox_inches="tight")
        #     plt.close()
        # plot_losses(avg_test_losses, avg_train_losses)

        # def plot_lin(model, dataloader):
        #     ys = []
        #     predicts = []
        #     for X, y in dataloader:
        #         pred = torch.atleast_2d(model(X.to(device))).squeeze(1).tolist()
        #         predicts += pred
        #         y = y.tolist()
        #         ys += y
            
        #     y_ticks, pred_avg, pred_stdev = stats(ys, predicts)
            
        #     def plot_ref_line():
        #         buffer = 0.05
        #         ticks = np.linspace(y_ticks[0]-buffer, y_ticks[-1]+buffer, 5)
        #         plt.plot(
        #             ticks, ticks,
        #             label="Ref. Line (Slope = 1)",
        #             color="grey",
        #             linewidth=0.5,
        #             zorder=0
        #         )
            
        #     plot_ref_line()
            
        #     plt.scatter(y_ticks, pred_avg, label="Results on Val. Set", color="firebrick", zorder=5, s=16)
        #     plt.errorbar(y_ticks, pred_avg, yerr=pred_stdev, fmt="none", elinewidth=0.5, capsize=0.5, color="black", label="Std. Dev.", zorder=10)
            
        #     plt.xlim(-2.25, 1.35)
        #     plt.ylim(-2.25, 1.35)

        #     plt.legend()
        #     plt.xlabel(r"Actual $\delta C_9$")
        #     plt.ylabel(r"Predicted $\delta C_9$")
            
        #     def make_title(level):
        #         result = r"$\mu$"
        #         if level == "gen":
        #             result += ", Generator" 
        #         elif level=="det":
        #             result += ", Detector"  
        #         result += f", Convolutional\nEpochs: {epochs}, Learn. Rate: {learning_rate}"
        #         result += f"\nBatch Size: {train_batch_size} (Train), {test_batch_size} (Test)"
        #         return result
            
        #     plt.title(make_title(level), loc="right")
            
        #     plt.savefig(f"../plots/cnn_lin_{level}.png", bbox_inches="tight")
        #     plt.close()
            
        # plot_lin(model, test_dataloader)


if __name__ == "__main__":
    main()