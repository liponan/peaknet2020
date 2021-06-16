import torch
import torch.nn as nn
import time

class AdaFilter_1(nn.Module):

    def __init__(self, params=None):
        super(AdaFilter_1, self).__init__()
        n_panels = 32
        self.n_panels = n_panels

        # Panel-dependent Filtering
        k_list = [3]
        n_list = []
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
                                               NL))
            self.pd_filtering = nn.Sequential(*layers)
        if self.adaptive_filtering:
            self.encoder = self.create_panel_to_filter_encoder()
            self.k_ada_filter = k_list[0]

        # Generic Peak Finding
        k_list = [3, 3]
        n_list = [3]
        NL = nn.LeakyReLU()
        self.residual = True
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

    def create_panel_to_filter_encoder(self):
        # h = 185 ~ 4 * 8 * 5, w = 388 ~ 8 * 8 * 6
        # k ** 2 = 9 -> 3, 6, 9 + 1 (bias) = 10
        NL = nn.ReLU()
        n_list = [3, 6, 10]
        k = 3
        #
        pad = (k - 1) // 2
        conv1 = nn.Conv2d(self.n_panels, n_list[0] * self.n_panels, k, padding=pad, groups=self.n_panels)
        pooling1 = nn.MaxPool2d([4, 8])
        conv2 = nn.Conv2d(n_list[0] * self.n_panels, n_list[1] * self.n_panels, k, padding=pad, groups=self.n_panels)
        pooling2 = nn.MaxPool2d([8, 8])
        conv3 = nn.Conv2d(n_list[1] * self.n_panels, n_list[2] * self.n_panels, k, padding=pad, groups=self.n_panels)
        pooling3 = nn.MaxPool2d([5, 6])
        encoder = nn.Sequential(conv1, NL, pooling1,
                                conv2, NL, pooling2,
                                conv3, NL, pooling3
                                )
        return encoder

    def use_encoder(self, x):
        # k = 3
        k = self.k_ada_filter
        N = x.size(0)
        filters_bias = self.encoder(x).view(N, self.n_panels, -1)
        filters = filters_bias[:, :, :-1].reshape(N * self.n_panels, 1, k, k)
        bias = filters_bias[:, :, -1:].reshape(-1)
        return filters, bias

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        if self.adaptive_filtering:
            filters, bias = self.use_encoder(x)
            # the filtering will be panel-dependent AND experiment-dependent
            x = x.view(1, -1, h, w)
            pad = (self.k_ada_filter - 1) // 2
            x = nn.ReflectionPad2d(pad)(x)
            filtered_x = nn.functional.conv2d(x, filters, bias=bias, groups=self.n_panels)
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