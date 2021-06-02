import torch
import torch.nn as nn

class AdaFilter_1(nn.Module):

    def __init__(self, params=None):
        super(AdaFilter_1, self).__init__()

        padding_mode = 'zeros'

        k_ada_filter_1 = 5
        in_ada_filter = 32
        out_ada_filter = 32
        groups_ada_filter = 32
        pad_ada_filter_1 = (k_ada_filter_1 - 1) // 2
        # conv_ada_filter_1 = nn.Sequential(nn.Conv2d(in_ada_filter, out_ada_filter, k_ada_filter_1,
        #                                                padding=pad_ada_filter_1,
        #                                                padding_mode=padding_mode,
        #                                                groups=groups_ada_filter),
        #                                      nn.BatchNorm2d(out_ada_filter),
        #                                      nn.ReLU())
        # k_ada_filter_2 = 5
        # pad_ada_filter_2 = (k_ada_filter_2 - 1) // 2
        # conv_ada_filter_2 = nn.Conv2d(out_ada_filter, out_ada_filter, k_ada_filter_2,
        #                                    padding=pad_ada_filter_2,
        #                                    padding_mode=padding_mode,
        #                                    groups=groups_ada_filter)
        # self.ada_filter = nn.Sequential(conv_ada_filter_1, conv_ada_filter_2)
        self.ada_filter = nn.Conv2d(in_ada_filter, out_ada_filter, k_ada_filter_1,
                                    padding=pad_ada_filter_1,
                                    padding_mode=padding_mode,
                                    groups=groups_ada_filter)
        k1 = 9
        in1 = 1
        out1 = 6
        pad1 = (k1 - 1) // 2
        conv1 = nn.Sequential(nn.Conv2d(in1, out1, k1, padding=pad1, padding_mode=padding_mode),
                                   nn.BatchNorm2d(out1),
                                   nn.ReLU())
        k2 = 3
        out2 = 1
        pad2 = (k2 - 1) // 2
        conv2 = nn.Conv2d(out1, out2, k2, padding=pad2, padding_mode=padding_mode)
        self.gen_peak_finding = nn.Sequential(conv1, conv2)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        filtered_x = self.ada_filter(x)
        filtered_x = filtered_x.view(-1, 1, h, w)
        logits = self.gen_peak_finding(filtered_x)
        return logits