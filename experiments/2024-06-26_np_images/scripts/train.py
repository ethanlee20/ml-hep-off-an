
"""
Train and evaluate a model.
"""

from torch.optim import Adam
from torch.nn import MSELoss

from analysis.modeling import train_test, lin_test, select_device
from analysis.modeling.datasets import Stacked_Hist2d_Dataset
from analysis.modeling.models import Stacked_Hist2d_Model


level = "gen"

learning_rate = 1e-4
epochs = 80
train_batch_size = 20
test_batch_size = 15

in_dir_path = "../datafiles/stacked_hist2d"

run_name = "test_test"

output_dirpath = "../models"

device = select_device()

model = Stacked_Hist2d_Model()

loss_fn = MSELoss()

optimizer = Adam(model.parameters(), lr=learning_rate)

dataset_train = Stacked_Hist2d_Dataset(
    in_dir_path, 
    "train", 
    level, 
    device
)

dataset_test = Stacked_Hist2d_Dataset(
    in_dir_path, 
    "test", 
    level, 
    device
)

train_test(
    model, 
    dataset_train, 
    dataset_test, 
    loss_fn, 
    optimizer, 
    device, 
    output_dirpath, 
    run_name, 
    epochs, 
    train_batch_size, 
    test_batch_size
)

lin_test(
    model, 
    dataset_test, 
    device, 
    output_dirpath, 
    run_name
)