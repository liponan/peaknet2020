import os
from glob import glob
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from data import PSANADataset, PSANAImage
from unet import UNet
from loss import PeaknetBCELoss
import visualize
import shutil
import argparse


def check_existence(exp, run):
    files = glob("/reg/d/psdm/cxi/{}/xtc/*{}*.xtc".format(exp, run))
    return len(files) > 0


def train(model, device, params):
    model.train()
    loss_func = PeaknetBCELoss(coor_scale=params["coor_scale"], pos_weight=params["pos_weight"]).to(device)
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
            x = x.view(-1, 1, h, w).to(device) # each panel is treated independently !!0
            y = y.view(-1, 3, h, w).to(device)
            scores = model(x)
            loss, recall, precision, rmsd = loss_func(scores, y, verbose=params["verbose"], cutoff=params["cutoff"])
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                seen += n
                print("seen {:6d}  loss {:7.5f}  recall  {:.3f}  precision {:.3f}  RMSD {:.3f}".
                      format(seen, float(loss.data.cpu()), recall, precision, rmsd))
                if seen % (params["backup_every"]) == 0:
                    torch.save(model.state_dict(), "debug/"+params["experiment_name"]+"/model.pt")
        psana_images.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("params", type=str, default=None, help="A json file")
    p.add_argument("--gpu", "-g", type=int, default=None, help="Use GPU x")
    p.add_argument("--model", "-m", type=str, default=None, help="A .PT file")
    return p.parse_args()


def main():
    args = parse_args()
    params = json.load(open(args.params))
    model = UNet(n_channels=1, n_classes=3, n_filters=params["n_filters"])
    if args.model:
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")
    model = model.to(device)

    model_dir = os.path.join('debug', params["experiment_name"])

    if os.path.exists(model_dir):
        val = str(input("The model directory %s exists. Overwrite? (y/n)" % model_dir))
        if val == 'y':
            shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    train(model, device, params)
    
    
if __name__ == "__main__":
    main()
