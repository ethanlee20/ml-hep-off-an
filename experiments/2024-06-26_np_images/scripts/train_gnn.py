
"""
Train and evaluate a model.
"""

from torch.optim import Adam
from torch.nn import MSELoss

from analysis.modeling import train_test, lin_test, select_device
from analysis.modeling.datasets import Gnn_Input_Dataset
from analysis.modeling.models import GravNet_Model


level = "gen"

learning_rate = 1e-4
epochs = 80
train_batch_size = 20
test_batch_size = 15

in_dir_path = "../datafiles/gnn_input"

run_name = "gnn_test"

output_dirpath = "../models"

device = select_device()

model = GravNet_Model(
    input_dim=4,
    output_dim=1,
    num_blocks=3,
    block_hidden_dim=20,
    block_output_dim=10,
    final_dense_dim=60,
    space_dim=3,
    prop_dim=15,
    k=6
)

loss_fn = MSELoss()

optimizer = Adam(model.parameters(), lr=learning_rate)

dataset_train = Gnn_Input_Dataset(
    in_dir_path, 
    level, 
    "train", 
    device
)

dataset_test = Gnn_Input_Dataset(
    in_dir_path, 
    level, 
    "test", 
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