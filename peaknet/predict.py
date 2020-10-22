import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from peaknet.data import PSANADataset, PSANAImage
from unet import UNet
from peaknet.loss import PeaknetBCELoss
from peaknet.train import check_existence
import argparse


def extract(scores, conf_cutoff=0.1):
    scores = nn.Sigmoid()(scores)
    scores_c = scores[:, 0, :, :].reshape(-1)
    conf_mask = scores_c > conf_cutoff
    scores_x = scores[:, 2, :, :].reshape(-1)[conf_mask]
    scores_y = scores[:, 1, :, :].reshape(-1)[conf_mask]
    uv = torch.nonzeros(conf_mask)
    predicted_x = scores_x + uv[:, 1].float()
    predicted_y = scores_y + uv[:, 0].float()
    output = torch.cat((predicted_x[:, None], predicted_y[:, None], scores_c[conf_mask, None]), dim=1)
    output = output.cpu().data.numpy()
    return output


def predict(model, device, params):
    model.eval()
    loss_func = PeaknetBCELoss().to(device)
    val_dataset = PSANADataset(params["run_dataset_path"], subset="val", shuffle=False)
    seen = 0
    acc_rec = 0
    acc_pre = 0
    acc_rms = 0
    acc_dt = 0
    for i, (cxi_path, exp, run) in enumerate(val_dataset):
        if check_existence(exp, run):
            pass
        else:
            print("[{:}] exp: {}  run: {}  PRECHECK FAILED".format(i, exp, run))
            continue
        print("*********************************************************************")
        print("[{:}] exp: {}  run: {}\ncxi: {}".format(i, exp, run, cxi_path))
        print("*********************************************************************")
        psana_images = PSANAImage(cxi_path, exp, run, downsample=params["downsample"], n=params["n_per_run"])
        data_loader = DataLoader(psana_images, batch_size=params["batch_size"], shuffle=True, drop_last=True,
                                 num_workers=params["num_workers"])
        for j, (x, y) in enumerate(data_loader):
            with torch.no_grad():
                n = x.size(0)
                h, w = x.size(2), x.size(3)
                x = x.view(-1, 1, h, w).to(device)
                y = y.view(-1, 3, h, w).to(device)
                t1 = time.time()
                scores = model(x)
                t2 = time.time()
                results = extract(scores, conf_cutoff=params["cutoff"])
                seen += n
                dt = t2 - t1
                # TODO(leeneil) output data to a static file


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=str, default=None, help="Path to a trained UNet model")
    p.add_argument("--gpu", "-g", type=int, default=None, help="Use GPU x")
    p.add_argument("--cutoff", "-c", type=float, default=0.5, help="Condifence threshold")
    p.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size")
    p.add_argument("--n_filters", type=int, default=32, help="Number of filters in UNet's first layer")
    p.add_argument("--n_per_run", "-n", type=int, default=-1, help="Number of images to sample from a run")
    p.add_argument("--plot", action="store_true", help="save output images in debug/")
    return p.parse_args()


def main():
    args = parse_args()
    params = {"run_dataset_path": "/reg/neh/home/liponan/peaknet2020/data/val.csv",
              "verbose": False, "lr": 0.01, "weight_decay": 1e-4, "cutoff": args.cutoff,
              "batch_size": args.batch_size, "num_workers": 0, "downsample": 1, "n_per_run": args.n_per_run}
    model = UNet(n_channels=1, n_classes=3, n_filters=args.n_filters)
    model.load_state_dict(torch.load(args.model))
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")
    model = model.to(device)
    predict(model, device, params)


if __name__ == "__main__":
    main()
