from torch import nn
import torch
from torch.nn import functional as F


class FCDiscriminator(nn.Module):
    ### This has been adapted directly from this repo:
    ### https://github.com/hfslyc/AdvSemiSeg

    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.gap   = nn.AdaptiveAvgPool2d(1)

        self.FC    = nn.Conv2d(in_channels=ndf*8, out_channels=2, kernel_size=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.gap(x)
        x = self.FC(x)
        return x