
from torch import nn


class Stacked_Hist2d_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 25, 5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(25, 50, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(50, 50, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(50, 100, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(100, 100, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2,1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dense = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(100, 25),
            # nn.Dropout(),
            nn.Linear(25, 1)
        )
        self.double()

    def forward(self, x):
        # breakpoint()
        z = self.conv(x)
        z = z.squeeze()
        result = self.dense(z)
        return result
