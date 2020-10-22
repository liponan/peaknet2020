import torch
from peaknet.predict import extract
from unet import UNet


class PeakNet(object):

    def __init__(self, n_filters=32):
        self.model = UNet(n_channels=1, n_classes=3, n_filters=n_filters)
        self.model_path = None
        self.n_filters = n_filters
        self.seen = 0

    def __repr__(self):
        msg = ("PeakNet\n# filters: {}\nPretrained model: {}".format(self.n_filters, self.model_path))
        return msg

    def to(self, device):
        self.model.to(device)

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model_path = model_path

    def predict(self, data, conf_cutoff=0.1):
        scores = self.model(data)
        output = extract(scores, conf_cutoff=conf_cutoff)
        return output

    def train(self, data):
        raise NotImplementedError("This method is not implemented yet.")

    def validate(self, data):
        raise NotImplementedError("This method is not implemented yet.")
