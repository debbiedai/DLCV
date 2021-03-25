import torch.nn as nn
import torch

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class Convnet(nn.Module):

    def __init__(self, in_channels=3, hid_channels=64, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, hid_channels),
            conv_block(hid_channels, out_channels),
        )


    def forward(self, x):
        x = self.encoder(x) #(b,64,5,5)
        x = x.view(x.size(0), -1)
        return x


class Hallucinator(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear0 = nn.Sequential(
            nn.Linear(2000, 1800),
            nn.ReLU(),
            nn.Linear(1800, 1600),
            nn.ReLU(),
        )
    def forward(self, x):
        x = self.linear0(x)
        return x

def euclidean_metric(a, b):
    logits = -((a - b)**2).sum(dim=2)
    return logits
class parametric(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.Sequential(
            nn.Linear(3200,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x, y):
        n = x.shape[0]
        m = y.shape[0]
        x = x.unsqueeze(1).expand(n, m, -1)
        y = y.unsqueeze(0).expand(n, m, -1)
        z = torch.cat((x,y), dim=2)
        z = self.linear0(z)
        z = z.squeeze()
        
        return z

