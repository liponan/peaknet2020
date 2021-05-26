import torch
import torch.nn as nn

class AdaFilter_0(nn.Module):

    def __init__(self, params=None, C=1):
        super(AdaFilter_0, self).__init__()
        self.C = C
        k1 = 11
        out1 = 6
        pad1 = (k1-1)//2
        conv1 = nn.Sequential(nn.Conv2d(C, out1, k1, padding=pad1, padding_mode='reflect'),
                                   nn.BatchNorm2d(out1),
                                   nn.ReLU())
        k2 = 3
        out2 = 1
        pad2 = (k2 - 1) // 2
        conv2 = nn.Conv2d(out1, out2, k2, padding=pad2, padding_mode='reflect')
        self.gen_peak_finding = nn.Sequential(conv1, conv2)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = x.view(-1, self.C, h, w)
        logits = self.gen_peak_finding(x)
        return logits