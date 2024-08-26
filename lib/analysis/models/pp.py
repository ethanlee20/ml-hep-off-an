

from torch import nn


class NN_pp(nn.Module):

    def __init__(self):
        
        super().__init__()

        self.stack = nn.Sequential(
                nn.Linear(4, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1)
        )
        
        self.double()

    def forward(self, x):
        
        x = self.stack(x)

        return x

