import torch
import torch.nn as nn
import time

class AdaFilter_1(nn.Module):

    def __init__(self, params=None):
        super(AdaFilter_1, self).__init__()
        n_panels = 32

        # Panel-dependent Filtering
        k_list = [3]
        n_list = []
        NL = nn.LeakyReLU()
        self.adaptive_filtering = True
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
        if self.adaptive_filtering:
            self.encoder = self.create_panel_to_filter_encoder()

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

    def create_panel_to_filter_encoder(self, k=3, n_panels=32):
        # h = 185 ~ 4 * 8 * 5, w = 388 ~ 8 * 8 * 6
        # k ** 2 = 9 -> 3, 6, 9 + 1 (bias) = 10
        NL = nn.ReLU()
        n_list = [3, 6, 10]
        conv1 = nn.Conv2d(n_panels, n_list[0] * n_panels, 3, padding=1, groups=n_panels)
        pooling1 = nn.MaxPool2d([4, 8])
        conv2 = nn.Conv2d(8 * n_panels, n_list[1] * n_panels, 3, padding=1, groups=n_panels)
        pooling2 = nn.MaxPool2d([8, 8])
        conv3 = nn.Conv2d(16 * n_panels, n_list[2] * n_panels, 3, padding=1, groups=n_panels)
        pooling3 = nn.MaxPool2d([5, 6])
        encoder = nn.Sequential(conv1, NL, pooling1,
                                conv2, NL, pooling2,
                                conv3, NL, pooling3
                                )
        return encoder

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        if self.adaptive_filtering:
            filters = self.encoder(x)
            print(filters.shape)
            print(self.state_dict().keys())
            print(self.state_dict()['pd_filtering.0.1.weight'].shape)
            print(self.state_dict()['pd_filtering.0.1.bias'].shape)
            time.sleep(5)
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