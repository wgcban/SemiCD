from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np
from utils.loss import CrossEntropy2d

class s4GAN_discriminator(nn.Module):
    '''
    https://github.com/sud0301/semisup-semseg/blob/13ec6f14ffe55cb47b01164ecf845d87f6904205/model/discriminator.py#L4 
    '''
    def __init__(self, num_classes, ndf = 64):
        super(s4GAN_discriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes+6, ndf, kernel_size=4, stride=2, padding=1) # 256 x 256
        self.conv2 = nn.Conv2d(  ndf, ndf*2, kernel_size=4, stride=2, padding=1) # 128 x 128
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1) # 64 x 64
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1) # 32 x 32
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ndf*8, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.drop = nn.Dropout2d(0.5)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
       
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
       
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.drop(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        maps = self.avgpool(x)
        conv4_maps = maps 
        out = maps.view(maps.size(0), -1)
        out = self.sigmoid(self.fc(out))
        
        return out, conv4_maps

def compute_argmax_map(output):
    output = output.detach().cpu().numpy()
    output = output.transpose((1,2,0))
    output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
    output = torch.from_numpy(output).float()
    return 

def find_good_maps(D_outs, pred_all):
    count = 0
    #Manual settings
    #Please check the batch 
    threshold_st = 0.6
    for i in range(D_outs.size(0)):
        if D_outs[i] > threshold_st:
            count +=1

    if count > 0:
        print ('Above ST-Threshold : ', count, '/', 8)
        pred_sel = torch.Tensor(count, pred_all.size(1), pred_all.size(2), pred_all.size(3))
        label_sel = torch.Tensor(count, pred_sel.size(2), pred_sel.size(3))
        num_sel = 0 
        for j in range(D_outs.size(0)):
            if D_outs[j] > threshold_st:
                pred_sel[num_sel] = pred_all[j]
                label_sel[num_sel] = compute_argmax_map(pred_all[j])
                num_sel +=1
        return  pred_sel.cuda(), label_sel.cuda(), count  
    else:
        return 0, 0, count 

def loss_calc(pred, label):
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d().cuda()  # Ignore label ??
    return criterion(pred, label)