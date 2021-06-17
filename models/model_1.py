import torch
import torch.nn as nn
import time

class AdaFilter_1(nn.Module):

    def __init__(self, params=None):
        super(AdaFilter_1, self).__init__()
        n_panels = 32
        self.n_panels = n_panels

        # Panel-dependent Filtering
        k_list = [3, 3]
        n_list = [16]
        NL = nn.LeakyReLU()
        self.adaptive_filtering = True
        #
        if not self.adaptive_filtering:
            in_list = [n_panels] + n_list
            out_list = n_list + [n_panels]
            pad_list = [(k - 1) // 2 for k in k_list]
            layers = []
            for i in range(len(k_list)):
                conv = nn.Conv2d(in_list[i], out_list[i], k_list[i], groups=n_panels)
                layers.append(nn.Sequential(nn.ReflectionPad2d(pad_list[i]),
                                            conv,
                                            torch.nn.GroupNorm(n_panels, out_list[i]),
                                            NL))
            self.pd_filtering = nn.Sequential(*layers)
        else:
            self.encoder, self.linear_layer = self.create_panel_to_filter_encoder()
            self.k_ada_filter = k_list[0]

        # Generic Peak Finding
        k_list = [3, 3]
        n_list = [3]
        NL_list = [nn.LeakyReLU(), nn.Tanh()]
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
                                           NL_list[i]))
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

    def create_panel_to_filter_encoder(self):
        # h = 185 ~ 4 * 8 * 5, w = 388 ~ 8 * 8 * 6
        # k ** 2 = 9 -> 3, 6, 9 + 1 (bias) = 10
        NL = nn.ReLU()
        n_list = [3, 6, 10]
        k = 3
        #
        pad = (k - 1) // 2
        conv1 = nn.Conv2d(self.n_panels, n_list[0] * self.n_panels, k, groups=self.n_panels, stride=(4, 8))
        norm1 = torch.nn.GroupNorm(self.n_panels, n_list[0] * self.n_panels)
        conv2 = nn.Conv2d(n_list[0] * self.n_panels, n_list[1] * self.n_panels, k, groups=self.n_panels, stride=(8, 8))
        norm2 = torch.nn.GroupNorm(self.n_panels, n_list[1] * self.n_panels)
        conv3 = nn.Conv2d(n_list[1] * self.n_panels, n_list[2] * self.n_panels, k, groups=self.n_panels, stride=(5, 6))
        norm3 = torch.nn.GroupNorm(self.n_panels, n_list[2] * self.n_panels)
        encoder = nn.Sequential(conv1, norm1, NL,
                                conv2, norm2, NL,
                                conv3, norm3, NL
                                )
        linear_layer = nn.Linear(n_list[-1], n_list[-1])
        return encoder, linear_layer

    def use_encoder(self, x):
        # k = 3
        k = self.k_ada_filter
        N = x.size(0)
        filters_bias = self.linear_layer(self.encoder(x).view(N * self.n_panels, -1))
        filters = filters_bias[:, :-1].view(N * self.n_panels, 1, k, k)
        bias = filters_bias[:, -1:].view(-1)
        return filters, bias

    def forward(self, x, return_intermediate_act=False):
        N, h, w = x.size(0), x.size(2), x.size(3)
        if self.adaptive_filtering:
            filters, bias = self.use_encoder(x)
            # the filtering will be panel-dependent AND experiment-dependent
            filtered_x = x.view(1, -1, h, w)
            pad = (self.k_ada_filter - 1) // 2
            filtered_x = nn.ReflectionPad2d(pad)(filtered_x)
            filtered_x = nn.functional.conv2d(filtered_x, filters, bias=bias, groups=N*self.n_panels)
        else:
            filtered_x = self.pd_filtering(x)
        # generic peak finiding is panel/experiment-independent
        filtered_x = filtered_x.view(-1, 1, h, w)
        logits = self.gen_peak_finding(filtered_x)
        if self.residual:
            logits += filtered_x
        # panel-dependent scaling
        panel_logits = logits.view(-1 , 32, h, w)
        panel_logits = self.pd_scaling(panel_logits)
        logits_out = panel_logits.view(-1, 1, h, w)
        if return_intermediate_act:
            return x, filtered_x, logits, logits_out
        else:
            return logits_out