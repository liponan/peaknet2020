"""
Full assembly of the parts to form the complete network
https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
"""

import torch.nn.functional as F
from .unet_parts import *
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters=64, bilinear=True, params=None):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, n_filters)
        self.down1 = Down(n_filters, n_filters*2)
        self.down2 = Down(n_filters*2, n_filters*4)
        self.down3 = Down(n_filters*4, n_filters*8)
        self.down4 = Down(n_filters*8, n_filters*8)
        self.up1 = Up(n_filters*16, n_filters*4, bilinear)
        self.up2 = Up(n_filters*8, n_filters*2, bilinear)
        self.up3 = Up(n_filters*4, n_filters, bilinear)
        self.up4 = Up(n_filters*2, n_filters, bilinear)
        self.outc = OutConv(n_filters, n_classes)

        self.downsample_bool = params["downsample"] >= 2
        # Downsampling
        if self.downsample_bool:
            self.downsampling = nn.MaxPool2d(params["downsample"])

        # Additional parameters that will be recovered when loading the model
        self.dataset_path = params["run_dataset_path"]
        self.downsample = params["downsample"]

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x = x.view(-1, 1, h, w)
        if self.downsample_bool:
            x_ds = self.downsampling(x)
        else:
            x_ds = x
        x1 = self.inc(x_ds)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def downsample_for_visualization(self, x):
        if self.downsample_bool:
            x_ds = self.downsampling(x)
        else:
            x_ds = x
        return x_ds