

from torch import nn


class TransformerRegressor(nn.Module):

    def __init__(self):
        
        super().__init__()

        self.first = nn.Linear(4, 4)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(4, 2, batch_first=True),
            3
        )

        self.last = nn.Linear(4, 1)

        self.double()

    def forward(self, x):
        # breakpoint()
        x = self.first(x)

        x = self.transformer_encoder(x)

        x = self.last(x)

        return x

