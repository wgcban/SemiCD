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

class RotationPredHead(nn.Module):
    def __init__(self, emb_dim, N_temp_rots):
        super(RotationPredHead, self).__init__()
        self.pool       = torch.nn.AdaptiveAvgPool2d((1))
        self.linear1    = torch.nn.Linear(2*emb_dim, emb_dim)
        self.relu       = torch.nn.ReLU()
        self.linear2    = torch.nn.Linear(emb_dim, N_temp_rots)

    def forward(self, z_a, z_b):
        z = torch.cat((torch.squeeze(self.pool(z_a)), torch.squeeze(self.pool(z_b))), dim=1)
        z = self.relu(self.linear1(z))
        r = self.linear2(z)
        return r

class RotationPredHeadSim(nn.Module):
    def __init__(self, emb_dim, N_temp_rots):
        super(RotationPredHeadSim, self).__init__()
        self.N          = 8
        self.pool       = torch.nn.AdaptiveAvgPool2d(self.N)
        self.softmax    = torch.nn.Softmax(dim=1)
        self.linear1    = torch.nn.Linear(self.N**4, 64)
        self.relu       = torch.nn.ReLU()
        self.linear2    = torch.nn.Linear(64, N_temp_rots)
        self.N_feat     = 16

    def forward(self, z_a, z_b):
        b, c, h , w = z_a.size()
        
        loc = torch.randint(c, (self.N_feat,))

        z_ar = z_a[:, loc, :, :]
        z_br = z_b[:, loc, :, :]

        z_ar = self.pool(z_ar).view(b, self.N_feat, self.N**2)
        z_br = self.pool(z_br).view(b, self.N_feat, self.N**2)

        C = self.softmax(torch.bmm(z_ar.permute(0, 2, 1), z_br))

        r = self.linear2(self.relu(self.linear1(C.view(b, self.N**4))))
        return r