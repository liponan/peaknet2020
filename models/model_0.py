import torch
import torch.nn as nn

class AdaFilter_0(nn.Module):

    def __init__(self, params=None, S=32, C=1):
        super(AdaFilter_0, self).__init__()
        self.S = S
        self.C = C

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x.view(-1, self.C, h, w)