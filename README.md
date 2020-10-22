# PeakNet 2020

Lighter, faster, better

## Usage

To create a PeakNet handle

```
from peaknet import PeakNet

net = PeakNet()
```


To use a pretrained model for prediction:

```
from peaknet import PeakNet

imgs = np.random.rand(1, 185, 388)
net = PeakNet(model_path=/cds/home/l/liponan/peaknet2020_old/debug/model.pt)
net.to("cuda:0")
output = net.predict(imgs, conf_cutoff=0.1)
```