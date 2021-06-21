import torch
import torch.nn as nn
import numpy as np
import time

class AdaFilter_1(nn.Module):

    def __init__(self, params=None):
        super(AdaFilter_1, self).__init__()
        n_panels = 32
        self.n_panels = n_panels
        self.can_show_inter_act = True
        self.downsample_bool = params["downsample"] >= 2

        # Downsampling
        if self.downsample_bool:
            self.downsampling = nn.MaxPool2d(params["downsample"])

        # Panel-dependent Filtering
        k_list = [3, 3]
        n_list = [16]
        NL = nn.LeakyReLU()
        self.adaptive_filtering = True
        self.adaptive_residual = False # keep False because each panel should be scaled differently
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
            h = 185 // params["downsample"]
            w = 388 // params["downsample"]
            self.encoder, self.linear_layer = self.create_panel_to_filter_encoder(k_list, n_list, h, w)
            self.k_ada_filter = k_list
            self.n_ada_filter = n_list

        # Generic Peak Finding
        k_list = [5, 5]
        n_list = [6]
        NL_list = [nn.ReLU()] * len(k_list)
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
                                        NL_list[i]))
        self.gen_peak_finding = nn.Sequential(*layers)
        if self.residual:
            self.combination_layer = nn.Conv2d(2, 1, 1)

        # Panel-Dependent Scaling
        k_list = []
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

    def create_panel_to_filter_encoder(self, k_list, n_list, h, w):
        # h = 185 ~ 4 * 8 * 5, w = 388 ~ 8 * 8 * 6
        # k ** 2 = 9 -> 3, 6, 9 + 1 (bias) = 10
        NL = nn.ReLU()
        k = 3
        n_layers = 3
        n_features = 8
        #
        # Build stride sizes
        stride_size_h = int(h ** (1 / n_features))
        stride_size_w = int(w ** (1 / n_features))
        s1_h, s2_h, s1_w, s2_w = stride_size_h, stride_size_h, stride_size_w, stride_size_w
        s3_h, s3_w = h // (stride_size_h ** 2), w // (stride_size_w ** 2)
        print("Stride Sizes:")
        print(h)
        print(s1_h, s2_h, s3_h)
        print(w)
        print(s1_w, s2_w, s3_w)
        #
        n_arr = np.array([1] + n_list + [1])
        k_arr = np.array(k_list)
        n_params = np.sum((n_arr[:-1] * k_arr ** 2 + 1) * n_arr[1:]) # total number of parameters for each panel
        channels_list = [n_features // (2 ** i) for i in range(n_layers)][::-1] # expand number of channels logarithmically
        conv1 = nn.Conv2d(self.n_panels, channels_list[0] * self.n_panels, k, groups=self.n_panels)
        pool1 = nn.AvgPool2d((s1_h, s1_w))
        norm1 = nn.GroupNorm(self.n_panels, channels_list[0] * self.n_panels)
        conv2 = nn.Conv2d(channels_list[0] * self.n_panels, channels_list[1] * self.n_panels, k, groups=self.n_panels)
        pool2 = nn.AvgPool2d((s2_h, s2_w))
        norm2 = nn.GroupNorm(self.n_panels, channels_list[1] * self.n_panels)
        pool3 = nn.AvgPool2d((s3_h, s3_w))
        conv3 = nn.Conv2d(channels_list[1] * self.n_panels, channels_list[2] * self.n_panels, k, groups=self.n_panels)
        norm3 = nn.GroupNorm(self.n_panels, channels_list[2] * self.n_panels)
        encoder = nn.Sequential(conv1, pool1, norm1, NL,
                                conv2, pool2, norm2, NL,
                                conv3, pool3, norm3, NL
                                )
        n_features_inter = int(np.sqrt(n_features * n_params))
        linear_layer = nn.Sequential(nn.Linear(n_features, n_features_inter),
                                     nn.LeakyReLU(),
                                     nn.Linear(n_features_inter, n_params)) # two-layer linear decoder
        return encoder, linear_layer

    def use_encoder(self, x, k_list, n_list):
        NL = nn.LeakyReLU()
        #
        N, h, w = x.size(0), x.size(2), x.size(3)
        # the filtering will be panel-dependent AND experiment-dependent
        filtered_x = x.view(1, -1, h, w)
        n_arr = np.array([1] + n_list + [1])
        k_arr = np.array(k_list)

        weight_bias = self.linear_layer(self.encoder(x).view(N * self.n_panels, -1))
        idx_beg = 0
        for i in range(len(k_list)):
            # Prepare filters
            n_weight = (n_arr[i] * k_arr[i] ** 2) * n_arr[i+1]
            n_bias = n_arr[i+1]
            weight = weight_bias[:, idx_beg:idx_beg+n_weight].reshape(N * self.n_panels * n_arr[i+1], n_arr[i], k_arr[i], k_arr[i])
            bias = weight_bias[:, idx_beg+n_weight:idx_beg+n_weight+n_bias].reshape(N * self.n_panels * n_arr[i+1],)
            idx_beg = idx_beg + n_weight + n_bias
            #
            pad = (k_arr[i] - 1) // 2
            filtered_x = nn.ReflectionPad2d(pad)(filtered_x)
            filtered_x = nn.functional.conv2d(filtered_x, weight, bias=bias, groups=N * self.n_panels)
            filtered_x = NL(filtered_x)
        return filtered_x

    def forward(self, x, return_intermediate_act=False):
        if self.downsample_bool:
            x_ds = self.downsampling(x)
        else:
            x_ds = x
        h, w = x_ds.size(2), x_ds.size(3)
        if self.adaptive_filtering:
            filtered_x = self.use_encoder(x_ds, self.k_ada_filter, self.n_ada_filter)
        else:
            filtered_x = self.pd_filtering(x_ds)
        # generic peak finding is panel/experiment-independent
        filtered_x = filtered_x.view(-1, 1, h, w)
        if self.adaptive_residual:
            filtered_x += x.view(-1, 1, h, w)
        logits = self.gen_peak_finding(filtered_x)
        if self.residual:
            logits = self.combination_layer(torch.cat((filtered_x, logits), 1))
        # panel-dependent scaling
        panel_logits = logits.view(-1 , 32, h, w)
        panel_logits = self.pd_scaling(panel_logits)
        logits_out = panel_logits.view(-1, 1, h, w)
        if return_intermediate_act:
            return x_ds, filtered_x, logits, logits_out
        else:
            return logits_out

    def downsample_for_visualization(self, x):
        if self.downsample_bool:
            x_ds = self.downsampling(x)
        else:
            x_ds = x
        return x_ds