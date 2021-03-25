import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url
# from torch.hub import load_state_dict_from_url
import torch.utils.model_zoo as model_zoo
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

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

class Conv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride, padding):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class VGG(nn.Module):
    def __init__(self, features, num_classes=7):
        super(VGG, self).__init__()
        self.features = features
        # self.conv1 = Conv(512,512,kernel_size=3,stride=1,padding=1)
        # # self.conv2 = Conv(256,256,kernel_size=3,stride=1,padding=1)
        # # self.conv3 = Conv(512,256,kernel_size=3,stride=1,padding=1)
        # self.transpose_conv1 = TransposeConv(512,256,kernel_size=4,stride=2,padding=1)
        # self.transpose_conv2 = TransposeConv(256,256,kernel_size=4,stride=2,padding=1)
        # self.transpose_conv3 = TransposeConv(256,256,kernel_size=4,stride=2,padding=1)
        # self.transpose_conv4 = TransposeConv(256,256,kernel_size=4,stride=2,padding=1)
        # self.transpose_conv5 = TransposeConv(256,7,kernel_size=4,stride=2,padding=1)


        self.fc1 = nn.Sequential(
            nn.Conv2d(512,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(512,1024,3,1,1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )
        
        self.fc3 = nn.Sequential(
            nn.Conv2d(1024,1024,1,1,0),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
        )
        self.pred1 = nn.Sequential(
            nn.ConvTranspose2d(1024,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,7,4,2,1),
        )

    def forward(self, x):
        #(b,3,32,32)
        x = self.features(x)
        # #(b,512,16,16)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.pred1(x)
        
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', False, pretrained, progress, **kwargs)


def model_segmentation(pretrained=True, **kwargs):
    vgg16_ = VGG(make_layers(cfgs['D'], batch_norm=False), **kwargs)
    if pretrained:
        vgg_state_dict = model_zoo.load_url(model_urls['vgg16'])
        vgg16_state_dict = vgg16_.state_dict()
        for k in vgg_state_dict.keys():
            if k.startswith('features'):
                vgg16_state_dict[k] = vgg_state_dict[k]
        vgg16_.load_state_dict(vgg16_state_dict)
    
    return vgg16_

