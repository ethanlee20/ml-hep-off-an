
from torch import nn


class Phi(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        self.dense = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        
        result = self.dense(x)
        
        return result
    

class Rho(nn.Module):
    
    def __init__(self):

        super().__init__()

        self.dense = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):

        result = self.dense(x)

        return result
    

class Deep_Sets(nn.Module):

    def __init__(self):

        super().__init__()

        self.phi = Phi()
        
        self.rho = Rho()

        self.double()

    def forward(self, x):
        
        phi_of_x = self.phi(x)
        
        sum_phi = phi_of_x.sum(axis=1)
        
        rho_of_sum = self.rho(sum_phi)
        
        # breakpoint()

        return rho_of_sum



