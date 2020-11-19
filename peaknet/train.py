import os
from glob import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data import PSANADataset, PSANAImage
from unet import UNet
from loss import PeaknetBCELoss
import argparse


def check_existence(exp, run):
    files = glob("/reg/d/psdm/cxi/{}/xtc/*{}*.xtc".format(exp, run))
    return len(files) > 0


def train(model, device, params):
    model.train()
    loss_func = PeaknetBCELoss(pos_weight=params["pos_weight"]).to(device)
    train_dataset = PSANADataset(params["run_dataset_path"], subset="train", shuffle=True)
    seen = 0
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    print("train_dataset", len(train_dataset))
    for i, (cxi_path, exp, run) in enumerate(train_dataset):
        #if os.path.isfile("good_cxi/{}_{}".format(exp, run)):
        #    pass
        #else:
        #    continue
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
            optimizer.zero_grad()
            n = x.size(0)
            h, w = x.size(2), x.size(3)
            x = x.view(-1, 1, h, w).to(device)
            y = y.view(-1, 3, h, w).to(device)
            scores = model(x)
            loss, recall, precision, rmsd = loss_func(scores, y, verbose=params["verbose"], cutoff=params["cutoff"])
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                seen += n
                print("seen {:6d}  loss {:7.5f}  recall  {:.3f}  precision {:.3f}  RMSD {:.3f}".
                      format(seen, float(loss.data.cpu()), recall, precision, rmsd))
                if seen % (100*params["batch_size"]) == 0:
                    torch.save(model.state_dict(), "debug/model.pt")
        psana_images.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", "-g", type=int, default=None, help="Use GPU x")
    p.add_argument("--batch_size", "-b", type=int, default=1, help="Batch size")
    p.add_argument("--n_filters", type=int, default=32, help="Number of filters in UNet's first layer")
    p.add_argument("--pos_weight", "-p", type=int, default=None, help="Weight for positive data")
    p.add_argument("--n_per_run", "-n", type=int, default=-1, help="Number of images to sample from a run")
    return p.parse_args()


def main():
    args = parse_args()
    params = {"run_dataset_path": "/reg/neh/home/liponan/peaknet2020/data/cxic0415.csv",
              "verbose": False, "lr": 0.01, "weight_decay": 1e-4, "cutoff": 0.2, "pos_weight": args.pos_weight,
              "batch_size": args.batch_size, "num_workers": 0, "downsample": 1, "n_per_run": args.n_per_run}
    model = UNet(n_channels=1, n_classes=3, n_filters=args.n_filters)
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")
    model = model.to(device)
    train(model, device, params)
    
    
if __name__ == "__main__":
    main()
