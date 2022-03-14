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
        self.pool = torch.nn.AdaptiveAvgPool2d((1))
        self.linear1 = torch.nn.Linear(2*emb_dim, emb_dim)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(emb_dim, N_temp_rots)

    def forward(self, z_a, z_b):
        z = torch.cat((torch.squeeze(self.pool(z_a)), torch.squeeze(self.pool(z_b))), dim=1)
        z = self.relu(self.linear1(z))
        r = self.linear2(z)
        return r