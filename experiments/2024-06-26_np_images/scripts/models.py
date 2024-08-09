
from torch import nn

class CompositeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "comp_net"
        self.avg_net = NeuralNetAvg()
        self.conv_1d_net = nn.Conv1d()
        self.conv_2d_net = nn.Conv2d()
        self.conv_3d_net = nn.Conv3d()


class NeuralNetAvg(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "nn_avg"
        self.stack = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )
        self.double()
    
    def forward(self, x):
        result = self.stack(x)
        return result


class Convolutional3D(nn.Module):
    def __init__(self, name):
        super().__init__()

        self.name = name

        self.conv = nn.Sequential(
            nn.Conv3d(6, 25, 4, padding="same"),
            nn.ReLU(),
            # nn.MaxPool3d(2, 1),
            nn.Conv3d(25, 50, 3, padding="same"),
            nn.ReLU(),
            nn.Conv3d(50, 50, 3, padding="same"),
            nn.ReLU(),
            # nn.MaxPool3d(2, 1),
            nn.Conv3d(50, 100, 3, padding="same"),
            nn.ReLU(),
            nn.Conv3d(100, 100, 3, padding="same"),
            nn.ReLU(),
            # nn.MaxPool3d(2,1),
            nn.AdaptiveAvgPool3d(1),
        )

        self.dense = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 8),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(8, 1)
        )

        self.double()

    def forward(self, x):
        z = self.conv(x)
        z = z.squeeze()
        result = self.dense(z)
        return result


class NN_Afb_S5(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(8, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, 125),
            nn.ReLU(),
            nn.Linear(125, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 1),
        )
        self.double()
    
    def forward(self, x):
        flat = self.flatten(x)
        result = self.stack(flat)
        return result



class Ensemble(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

        self.conv3d = Convolutional3D("cube_viewer")
        self.nn = NN_Afb_S5("AfbS5_viewer")

        self.top = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(10, 5),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(5,1)
        )

        self.double()
    
    def forward(self, x):
        out_nn = self.nn(x["afb_s5"])
        out_cnn = self.conv3d(x["cube"])
        breakpoint()
        result = self.top()




class Convolutional2D(nn.Module):
    def __init__(self, name):
        super().__init__()
        
        self.name = name

        self.conv = nn.Sequential(
            nn.Conv2d(10, 25, 5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(25, 50, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(50, 50, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(50, 100, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(100, 100, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dense = nn.Sequential(
            nn.Dropout(),
            nn.Linear(100, 25),
            nn.Dropout(),
            nn.Linear(25, 1)
        )
        self.double()

    def forward(self, x):
        z = self.conv(x)
        z = z.squeeze()
        result = self.dense(z)
        return result


class NeuralNetwork(nn.Module):
    def __init__(self, name):
        super().__init__()
        
        self.name = name
        
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
