import torch
import torch.nn as nn

class AdaFilter_1(nn.Module):

    def __init__(self, params=None):
        super(AdaFilter_1, self).__init__()

        padding_mode = 'zeros'

        k_ada_filter_1 = 1
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
                                    groups=groups_ada_filter,
                                    bias=False)

        k_list = [3, 3, 3, 3]
        in_list = [1, 6, 6, 6]
        out_list = in_list[1:] + [6]
        pad_list = [(k - 1) // 2 for k in k_list]
        conv_list = []

        for i in range(len(k_list)):
            conv_list.append(nn.Sequential(nn.Conv2d(in_list[i], out_list[i], k_list[i], padding=pad_list[i], padding_mode=padding_mode),
                                           nn.BatchNorm2d(out_list[i]),
                                           nn.ReLU()))

        k_out = 1
        pad_out = (k_out - 1) // 2
        conv_out = nn.Conv2d(out_list[-1], 1, k_out, padding=pad_out, padding_mode=padding_mode)
        conv_list.append(conv_out)
        self.gen_peak_finding = nn.Sequential(*conv_list)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        filtered_x = self.ada_filter(x)
        filtered_x = filtered_x.view(-1, 1, h, w)
        logits = self.gen_peak_finding(filtered_x)
        return logits