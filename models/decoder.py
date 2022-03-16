import math , time
import torch
import torch.nn.functional as F
from torch import nn
from utils.helpers import initialize_weights
from itertools import chain
import contextlib
import random
import numpy as np
import cv2
from torch.distributions.uniform import Uniform


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """
    Checkerboard artifact free sub-pixel convolution
    https://arxiv.org/abs/1707.02937
    """
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)


class PixelShuffle(nn.Module):
    """
    Real-Time Single Image and Video Super-Resolution
    https://arxiv.org/abs/1609.05158
    """
    def __init__(self, n_channels, scale):
        super(PixelShuffle, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels*(scale**2), kernel_size=1)
        icnr(self.conv.weight)
        self.shuf = nn.PixelShuffle(scale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.shuf(self.relu(self.conv(x)))
        return x


# def upsample(in_channels, out_channels, upscale, kernel_size=3):
#     # A series of x 2 upsamling until we get to the upscale we want
#     layers = []
#     conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#     nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
#     layers.append(conv1x1)
#     for i in range(int(math.log(upscale, 2))):
#         layers.append(PixelShuffle(out_channels, scale=2))
#     return nn.Sequential(*layers)

def upsample(in_channels, out_channels, upscale, kernel_size=3):
    layers = []

    # Middle channels
    mid_channels = 32

    #First conv layer to reduce number of channels
    diff_conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=False)
    nn.init.kaiming_normal_(diff_conv1x1.weight.data, nonlinearity='relu')
    layers.append(diff_conv1x1)

    #ReLU
    diff_relu = nn.ReLU()
    layers.append(diff_relu)

    #Upsampling to original size
    up      = nn.Upsample(scale_factor=upscale, mode='bilinear')
    layers.append(up)

    #Classification layer
    conv1x1 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(conv1x1.weight.data, nonlinearity='relu')
    layers.append(conv1x1)

    return nn.Sequential(*layers)


class PixShuffleDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(PixShuffleDecoder, self).__init__()
        print(upscale)
        self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

    def forward(self, x):
        x = self.upsample(x)
        return x


# Convolutional Decoder
class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                            padding=kernel_size // 2, stride=1)
                         )

class ConvDecoder(nn.Module):
    def __init__(self, upscale, conv_in_ch, num_classes):
        super(ConvDecoder, self).__init__()
        print(upscale)
        self.diff_conv  = TwoLayerConv2d(in_channels=conv_in_ch, out_channels=32)
        self.upsample   = nn.Upsample(scale_factor=upscale, mode='bilinear')
        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=num_classes)

    def forward(self, x):
        x = self.diff_conv(x)
        x = self.upsample(x)
        x = self.classifier(x)
        return x