
"""
Train and evaluate a DeepSets model.
"""

from torch.optim import Adam
from torch.nn import MSELoss

from torch.profiler import profile, record_function, ProfilerActivity

from analysis.modeling import train_test, lin_test, select_device
from analysis.datasets import Gnn_Input_Dataset
from analysis.modeling.models import Deep_Sets
from analysis.plot import setup_mpl_params
setup_mpl_params()


level = "det"

learning_rate = 6e-4
epochs = 50
train_batch_size = 32
test_batch_size = 16

in_dir_path = "../datafiles/gnn_input"

run_name = "deepsets_det"

output_dirpath = "../models"

device = select_device()

model = Deep_Sets()

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