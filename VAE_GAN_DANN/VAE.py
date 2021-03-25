import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.conv1 = Conv(3, 64, 3, 2, 1)
        self.conv2 = Conv(64, 128, 3, 2, 1)
        self.maxpool = nn.MaxPool2d((2,2))
        self.fc1 = nn.Linear(128*8*8, 1024)
        self.fc21 = nn.Linear(1024, 512)
        self.fc22 = nn.Linear(1024, 512)

        self.conv3 = Conv(32,64,3,1,1)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv4 = Conv(64,128,3,1,1)
        self.upsample2 = nn.Upsample(scale_factor=2)    #(128,16,16)
        self.conv5 = Conv(128,128,3,1,1)
        self.maxpool2 = nn.MaxPool2d((2,2))
        self.fc3 = nn.Linear(128*8*8,1024)
        self.fc4 = nn.Linear(1024,64*64*3)


        # self.fc3 = nn.Linear(512, 1024)
        # self.fc4 = nn.Linear(1024, 2048)
        # self.fc5 = nn.Linear(2048, 4096)
        # self.fc6 = nn.Linear(4096, 64*64*3)

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        h1 = F.relu(x)
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()    #(b,512)
        eps = torch.cuda.FloatTensor(std.size()).normal_()  #(b,512)
        a = eps.mul(std).add_(mu)
        return eps.mul(std).add_(mu)    #(b,512)

    def decode(self, z):
        #(b,512)
        batch = z.size(0)
        z = z.view((batch,32,4,4)) 
        z = self.conv3(z)
        z = self.upsample1(z)
        z = self.conv4(z)
        z = self.upsample2(z)
        z = self.conv5(z)
        z = self.maxpool2(z)    #(b,256,8,8)
        z = torch.flatten(z,1)
        z = self.fc3(z)
        z = self.fc4(z)
        return z


    def forward(self, x, mode):
        # (b,3,64,64)
        if mode == 'test':
            recon = self.decode(x)
            return recon
        else:
            mu, logvar = self.encode(x)
            z = self.reparametrize(mu, logvar)  #(b,512)
            output = self.decode(z)
            return output, mu, logvar, z
            
