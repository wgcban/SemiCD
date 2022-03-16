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
from torchvision import transforms


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
        self.N_temp_rots = N_temp_rots

        self.N          = 8

        self.pool       = torch.nn.AdaptiveAvgPool2d(self.N)
        self.softmax    = torch.nn.Softmax(dim=1)
        self.linear1    = torch.nn.Linear(self.N**4, 64)
        self.relu       = torch.nn.ReLU()
        self.linear2    = torch.nn.Linear(64, N_temp_rots)
        self.N_feat     = 8

    def forward(self, z_a, z_b, cm, target_l_r):
        b, c, h , w = z_a.size()

        #Get the actual angle and rotate the predicted mask according to that
        angle   = 360*target_l_r/self.N_temp_rots

        cm_r = cm.unsqueeze(1)
        for i in range(b):
            cm_r[i] = transforms.functional.rotate(cm[i].unsqueeze(0), angle=angle[i].item(), fill=0.0)
        
        #Apply change mask on
        z_a     = z_a*(1.0-torch.nn.functional.interpolate(cm_r, size=[h,w], mode='nearest'))
        z_b     = z_b*(1.0-torch.nn.functional.interpolate(cm_r, size=[h,w], mode='nearest'))
        
        loc = torch.randint(c, (self.N_feat,))

        z_ar = z_a[:, loc, :, :]
        z_br = z_b[:, loc, :, :]

        z_ar = self.pool(z_ar).view(b, self.N_feat, self.N**2)
        z_br = self.pool(z_br).view(b, self.N_feat, self.N**2)

        C = self.softmax(torch.bmm(z_ar.permute(0, 2, 1), z_br))

        r = self.linear2(self.relu(self.linear1(C.view(b, self.N**4))))
        return r