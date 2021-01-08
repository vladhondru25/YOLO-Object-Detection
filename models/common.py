import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super(Mish,self).__init__()

    def forward(self, x):
        return x * nn.tanh(nn.softplus(x))


ACTIVATIONS = {
    'mish': Mish(),
    'relu': nn.ReLU(inplace=True),
    'leakyRelu': nn.LeakyReLU(negative_slope=0.01, inplace=True),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'identity': nn.Identity()
}


class Conv2dBlock(nn.Module):
    def __init__(self, in_C, out_C, k, s, p, bias=False, activation='leakyRelu'):
        super(Conv2dBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_C, out_channels=out_C, kernel_size=k, stride=s, padding=p, bias=bias),
            nn.BatchNorm2d(out_C),
            ACTIVATIONS[activation]
        )

    def forward(self, x):
        return self.conv_layer(x)


class BottleneckBlock(nn.Module):
    def __init__(self, in_C, activation='leakyRelu'):
        super(BottleneckBlock, self).__init__()

        self.conv1 = Conv2dBlock(in_C=in_C,    out_C=in_C//2, k=1, s=1, p=0)
        self.conv2 = Conv2dBlock(in_C=in_C//2, out_C=in_C,    k=3, s=1, p=1, activation='identity')

        self.activation = ACTIVATIONS[activation]

    def forward(self, x):
        z1 = self.conv1(x)
        z2 = x + self.conv2(z1)
        return self.activation(z2)


class CSPBlock(nn.Module):
    def __init__(self, in_C, no_of_blocks=1, activation='leakyRelu'):
        super(CSPBlock, self).__init__()
        hidden_channels = in_C//2

        self.conv_left  = nn.Conv2d(in_channels=in_C, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.right_block = nn.Sequential(
            Conv2dBlock(in_C=in_C, out_C=hidden_channels, k=1, s=1, p=0, activation=activation),
            *[BottleneckBlock(hidden_channels) for _ in range(no_of_blocks)],
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.common_conv = nn.Sequential(
            nn.BatchNorm2d(in_C),
            ACTIVATIONS[activation],
            Conv2dBlock(in_C=in_C, out_C=in_C, k=1, s=1, p=0, activation=activation)
        )

    def forward(self, x):
        left  = self.conv_left(x)
        right = self.right_block(x)

        return self.common_conv(torch.cat([left,right], dim=1))


class SPP(nn.Module):
    def __init__(self, in_C, out_C, kernels=[5,9,13]):
        super(SPP, self).__init__()
        
        self.cv1 = Conv2dBlock(in_C=in_C, out_C=in_C//2, k=1, s=1, p=0)
        
        self.spatial_pyramid = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2) for k in kernels]
        )
        
        self.cv2 = Conv2dBlock(in_C=in_C*2, out_C=out_C, k=1, s=1, p=0)

    def forward(self, x):
        x = self.cv1(x)
        
        spatial_pyramid_output = torch.cat([x] + [level(x) for level in self.spatial_pyramid], dim=1)
        
        return self.cv2(spatial_pyramid_output)