from unet import UNet


class PeakNet(object):

    def __init__(self, n_filters=32):
        self.model = UNet(n_channels=1, n_classes=3, n_filters=n_filters)
        self.seen = 0

    def use_device(self, device):
        self.model.to(device)