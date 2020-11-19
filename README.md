# PeakNet 2020

Lighter, faster, better -- Peak finding for LCLS crystallography

Author: Po-Nan Li @ Stanford & SLAC


## Setup

Switch to `ana-1.4.22`

```
conda activate ana-1.4.22
```

Install package locally

```
cd peaknet2020
mkdir -p install/lib/python2.7/site-packages
export PYTHONPATH=`pwd`/install/lib/python2.7/site-packages
python setup.py develop --prefix=`pwd`/install
```

## Usage

To create a PeakNet handle

```
from peaknet import PeakNet

net = PeakNet()
```


To use a pretrained model for prediction:

```
import torch
import numpy as np
from peaknet import PeakNet

imgs = torch.from_numpy(np.random.rand(1, 1, 185, 388)).to("cuda:0")
net = PeakNet(model_path="/cds/home/l/liponan/peaknet2020_old/debug/model.pt")
net.to("cuda:0")
output = net.predict(imgs.float(), conf_cutoff=0.1)
```

To train a model from scratch

```
cd peaknet
python train.py param.json -g 0
```

`-g 0` specifies to use GPU 0 on the machine.


## Credits

PyTorch model of UNet is due to https://github.com/milesial/Pytorch-UNet
