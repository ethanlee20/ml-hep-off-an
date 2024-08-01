
from torch import nn


class ConvolutionalNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 25, 5, padding="same"),
            nn.ReLU(),
            nn.MaxPool3d(2,2),
            nn.Conv3d(25, 50, 3, padding="same"),
            nn.ReLU(),
            nn.Conv3d(50, 50, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool3d(2, 2),
            nn.Conv3d(50, 100, 3, padding="same"),
            nn.ReLU(),
            nn.Conv3d(100, 100, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool3d(2,2),
            nn.AdaptiveAvgPool3d(1),
        )

        self.dense = nn.Sequential(
            nn.Dropout(),
            nn.Linear(100, 25),
            nn.Dropout(),
            nn.Linear(25, 1)
        )

        self.double()

    def forward(self, x):
        x = x.unsqueeze(1)
        z = self.conv(x)
        z = z.squeeze()
        result = self.dense(z)
        return result
