import torch
import torch.nn as nn

class AdaFilter_1(nn.Module):

    def __init__(self, params=None):
        super(AdaFilter_1, self).__init__()

        k_ada_filter_1 = 1
        in_ada_filter = 32
        out_ada_filter = 32
        groups_ada_filter = 32
        pad_ada_filter_1 = (k_ada_filter_1 - 1) // 2
        conv = nn.Conv2d(in_ada_filter, out_ada_filter, k_ada_filter_1,
                         padding=0,
                         groups=groups_ada_filter,
                         bias=False)
        self.ada_filter = nn.Sequential(nn.ReflectionPad2d(pad_ada_filter_1), conv) # input in [0, 1], output in [0, ?]
        # torch.nn.init.xavier_uniform_(conv.weight)

        k_list = [5, 5]
        in_list = [1, 3]
        out_list = in_list[1:] + [1]
        pad_list = [(k - 1) // 2 for k in k_list]
        conv_list = []
        for i in range(len(k_list)):
            conv = nn.Conv2d(in_list[i], out_list[i], k_list[i], padding=0)
            conv_list.append(nn.Sequential(nn.ReflectionPad2d(pad_list[i]),
                                           conv,
                                           nn.BatchNorm2d(out_list[i]),
                                           nn.Tanh())) # symmetric output
            # torch.nn.init.xavier_uniform_(conv.weight)
        self.gen_peak_finding = nn.Sequential(*conv_list)

        k_out = 3
        pad_out = (k_out - 1) // 2
        conv_out = nn.Conv2d(32, 32, k_out, padding=0, groups=32)
        # torch.nn.init.xavier_uniform_(conv_out.weight)
        self.conv_out = nn.Sequential(nn.ReflectionPad2d(pad_out), conv_out)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        filtered_x = self.ada_filter(x)
        filtered_x = filtered_x.view(-1, 1, h, w)
        logits = self.gen_peak_finding(filtered_x)
        panel_logits = logits.view(-1 , 32, h, w)
        panel_logits = self.conv_out(panel_logits)
        logits_out = panel_logits.view(-1, 1, h, w)
        return logits_out