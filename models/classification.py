import torch
import torch.nn as nn
import torch.nn.functional as F

class flood_classify(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(flood_classify, self).__init__()
        self.input_fc = nn.Linear(input_dim, 256)
        self.hidden_fc = nn.Linear(256, 128)
        self.output_fc = nn.Linear(128, output_dim)

    def forward(self, x):
        #x = [batch size, c, height, width]
        batch_size = x.shape[0]

        # Global average pooling layer
        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)))

        #x = [batch size, c]
        x = x.view(batch_size, -1)

        #h_1 = [batch size, 256]
        h_1 = F.relu(self.input_fc(x))

        #h_2 = [batch size, 128]
        h_2 = F.relu(self.hidden_fc(h_1))

        #y_pred = [batch size, output dim]
        y_pred = self.output_fc(h_2)
        return y_pred