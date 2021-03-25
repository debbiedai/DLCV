import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

class TransposeConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding):
        super(TransposeConv, self).__init__()
        self.transpose = nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self,x):
        x = self.transpose(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # self.linear1 = nn.Sequential(
        # nn.Linear(128,512),
        # # nn.BatchNorm1d(512),
        # nn.ReLU(True)
        # )

        #self.conv1 = Conv(8,64,3,2,1)
        #self.conv2 = Conv(64,128,3,1,1)

        self.trans1 = TransposeConv(128,256,4,1,0)
        self.trans2 = TransposeConv(256,512,4,2,1)
        self.trans3 = TransposeConv(512,256,4,2,1)
        self.trans4 = TransposeConv(256,128,4,2,1)
        # self.trans1 = TransposeConv(128,256,4,1,0)
        # self.trans2 = TransposeConv(256,512,4,2,1)
        # self.trans3 = TransposeConv(512,256,4,2,1)
        # self.trans4 = TransposeConv(256,128,4,2,1)
        # self.trans5 = TransposeConv(128,3,4,2,1)
        
        self.trans5 = nn.Sequential(
            nn.ConvTranspose2d(128,3,4,2,1),
            # nn.BatchNorm1d(3),
            nn.Tanh()
        )
        # self.conv1 = Conv(3,64,3,2,1)
        # self.conv2 = Conv(64,128,3,2,1)
        # self.conv3 = Conv(128,256,3,2,1)
        # self.conv4 = Conv(256,512,3,2,1)
        # self.main = nn.Sequential(
        #     nn.Linear(4*4*512, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 64*64*3),
        #     nn.Tanh()
        # )
        
        # self.main = nn.Sequential(
        #     nn.Linear(64*64*3, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 64*64*3),
        #     nn.Tanh()
        # )
    def forward(self, x):
        #(b,3,64,64)
        batch = x.size(0)
        # x = self.linear1(x)
        x = x.view(batch,128,1,1)
        # x = self.linear1(x)
        
        
        #x = self.conv1(x)
        #x = self.conv2(x)
        # x = self.conv3(x)
        x = self.trans1(x)
        x = self.trans2(x)
        x = self.trans3(x)
        x = self.trans4(x)
        x = self.trans5(x)
        #x = self.trans6(x)
        return x


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()

#         self.model = nn.Sequential(
#             nn.Linear(64*64*3, 512),

#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#             nn.Sigmoid(),
#         )

#     def forward(self, x):
#         x = torch.flatten(x,1)
#         x = self.model(x)
#         return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        self.pred = nn.Sequential(
            nn.Linear(16 * 16 * 64, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)

        x = torch.flatten(x,1)

        x = self.pred(x)
        return x