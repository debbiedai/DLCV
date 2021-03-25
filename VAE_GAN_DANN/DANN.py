import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

# class GradReverse(torch.autograd.Function):
#     """
#     Extension of grad reverse layer
#     """
#     @staticmethod
#     def forward(self, x, constant):
#         self.constant = constant
#         return x.view_as(x)

#     @staticmethod
#     def backward(self, grad_output):
#         grad_output = grad_output.neg() * self.constant
#         return grad_output, None

#     # def grad_reverse(x, constant):
#     #     return GradReverse.apply(x, constant)

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

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.linear = nn.Sequential(
            nn.Linear(128 * 7 * 7, 2048),
            
            nn.ReLU(True)
        )
        

    def forward(self, x):
        b = x.size(0)
        x = self.extractor(x)
        x = x.view(b, -1)
        x = self.linear(x)
        return x
    #     self.conv1 = Conv(3,32,3,1,1)
    #     self.maxpool1 = nn.MaxPool2d(2,2)
    #     self.conv2 = Conv(32,64,3,1,1)
    #     self.maxpool2 = nn.MaxPool2d(2,2)
    #     self.conv3 = Conv(64,128,3,1,1)
    #     # self.conv4 = Conv(128,256,3,1,1)
    #     self.linear1 = nn.Sequential(
    #         nn.Linear(128*7*7,2048),
    
    #         nn.ReLU(True)
    #     )


    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.maxpool1(x)
    #     x = self.conv2(x)
    #     x = self.maxpool2(x)
    #     x = self.conv3(x)
    #     # x = self.conv4(x)
    #     x = torch.flatten(x,1)
    #     x = self.linear1(x)
    #     return x



class Class_classifier(nn.Module):

    def __init__(self):
        super(Class_classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=10)
        )

    def forward(self, x):
        x = self.classifier(x)
        return F.log_softmax(x,1)
    #     self.linear1 = nn.Sequential(
    #         nn.Linear(2048,512),
    
    #         nn.ReLU()
    #     )
    #     self.linear2 = nn.Sequential(
    #         nn.Linear(512,10),
    #     )
    #     # self.linear1 = nn.Linear(2048, 512)
    #     # self.linear2 = nn.Linear(512,10)
    # def forward(self, x):
    #     # x = torch.flatten(x,1)
    #     x = self.linear1(x)
    #     x = self.linear2(x)
    #     x = F.log_softmax(x, 1)
    #     return  x

class Domain_classifier(nn.Module):

    def __init__(self):
        super(Domain_classifier, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            
            nn.ReLU(True),
            nn.Linear(in_features=1024, out_features=256),
            
            nn.ReLU(True),
            nn.Linear(256, 2)
        )

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.discriminator(reversed_input)
        logits = F.log_softmax(x, 1)
        return logits
    #     self.fc1 = nn.Sequential(
    #         nn.Linear(2048,1024),
    
    #         nn.ReLU()
    #     )
    #     self.fc2 = nn.Sequential(
    #         nn.Linear(1024,512),
    
    #         nn.ReLU()
    #     )
    #     self.fc3 = nn.Linear(512, 2)
    #     # self.reverse =  ReverseLayerF()

    # def forward(self, x, constant):
    #     #(b,3,28,28)
    #     x = ReverseLayerF.apply(x, constant)
    #     x = self.fc1(x)
    #     x = self.fc2(x)
    #     x = self.fc3(x)
    #     logits = F.log_softmax(x, 1)
    #     # x = torch.flatten(x,1)
    #     # logits = F.relu(self.fc1(x))
    #     # logits = self.fc2(logits)
    #     # logits = F.log_softmax(logits, 1)

    #     return logits

