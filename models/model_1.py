import torch
import torch.nn as nn

class AdaFilter_1(nn.Module):

    def __init__(self, params=None):
        super(AdaFilter_1, self).__init__()
        n_panels = 32

        # Panel-dependent Filtering
        k_list = [3, 3]
        n_list = [4 * n_panels]
        NL = nn.LeakyReLU()
        #
        in_list = [n_panels] + n_list
        out_list = n_list + [n_panels]
        pad_list = [(k - 1) // 2 for k in k_list]
        layers = []
        for i in range(len(k_list)):
            conv = nn.Conv2d(in_list[i], out_list[i], k_list[i], groups=n_panels)
            layers.append(nn.Sequential(nn.ReflectionPad2d(pad_list[i]),
                                           conv,
                                           NL))
        self.pd_filtering = nn.Sequential(*layers)

        # Generic Peak Finding
        k_list = [5, 5]
        n_list = [3]
        NL = nn.Tanh()
        self.residual = False
        #
        in_list = [1] + n_list
        out_list = n_list + [1]
        pad_list = [(k - 1) // 2 for k in k_list]
        layers = []
        for i in range(len(k_list)):
            conv = nn.Conv2d(in_list[i], out_list[i], k_list[i])
            layers.append(nn.Sequential(nn.ReflectionPad2d(pad_list[i]),
                                           conv,
                                           nn.BatchNorm2d(out_list[i]),
                                           NL))
        self.gen_peak_finding = nn.Sequential(*layers)

        # Panel-Dependent Scaling
        k_list = [1]
        n_list = []
        #
        in_list = [n_panels] + n_list
        out_list = n_list + [n_panels]
        pad_list = [(k - 1) // 2 for k in k_list]
        layers = []
        for i in range(len(k_list)):
            conv = nn.Conv2d(in_list[i], out_list[i], k_list[i], groups=n_panels)
            layers.append(nn.Sequential(nn.ReflectionPad2d(pad_list[i]),
                                        conv))
        self.pd_scaling = nn.Sequential(*layers)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        filtered_x = self.pd_filtering(x)
        filtered_x = filtered_x.view(-1, 1, h, w)
        logits = self.gen_peak_finding(filtered_x)
        if self.residual:
            logits += filtered_x
        panel_logits = logits.view(-1 , 32, h, w)
        panel_logits = self.pd_scaling(panel_logits)
        logits_out = panel_logits.view(-1, 1, h, w)
        return logits_out

    def forward_with_inter_act(self, x):
        h, w = x.size(2), x.size(3)
        filtered_x = self.pd_filtering(x)
        filtered_x = filtered_x.view(-1, 1, h, w)
        logits = self.gen_peak_finding(filtered_x)
        if self.residual:
            logits += filtered_x
        panel_logits = logits.view(-1 , 32, h, w)
        panel_logits = self.pd_scaling(panel_logits)
        logits_out = panel_logits.view(-1, 1, h, w)
        return x, filtered_x, logits, logits_out