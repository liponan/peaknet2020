import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from peaknet.data import PSANADataset, PSANAImage
from unet import UNet
from peaknet.loss import PeaknetBCELoss
from peaknet.train import check_existence
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import argparse


def plot(x, y, scores, output_path, save_npy=True):
    x = x.data.cpu().numpy()
    y = y.data.cpu().numpy()
    scores = nn.Sigmoid()(scores.data).cpu().numpy()
    for i in range(x.shape[0]):
        if np.sum(y[i, 0, :, :]) < 1:
            continue
        fig = plt.figure()
        plt.subplot(3, 1, 1)
        plt.imshow(x[i, 0, :, :])
        plt.subplot(3, 1, 2)
        plt.imshow(y[i, 0, :, :], vmin=0, vmax=1)
        plt.subplot(3, 1, 3)
        plt.imshow(scores[i, 0, :, :], vmin=0, vmax=1)
        plt.savefig(output_path+"_x_y_scores{}.png".format(str(i).zfill(3)))
        plt.close()
        np.save(output_path + "_x{}.npy".format(str(i).zfill(3)), x[i, 0, :, :])
        np.save(output_path + "_y{}.npy".format(str(i).zfill(3)), y[i, :, :, :])
        np.save(output_path + "_scores{}.npy".format(str(i).zfill(3)), scores[i, :, :, :])


def validate(model, device, params, save_plot=False):
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
                loss, recall, precision, rmsd = loss_func(scores, y, verbose=params["verbose"], cutoff=params["cutoff"])
                seen += n
                dt = t2 - t1
                print("loss {:7.5f}  recall  {:.3f}  precision {:.3f}  RMSD {:.3f} in {:5.3f} ms".
                      format(float(loss.data.cpu()), recall, precision, rmsd, 1000*dt))
                acc_rec += n * recall
                acc_pre += n * precision
                acc_rms += n * rmsd
                acc_dt += n * dt
                if save_plot:
                    output_filename = "debug/val_{}_{}_{}".format(exp, run, str(j).zfill(6))
                    plot(x, y, scores, output_filename)
        psana_images.close()
    acc_rec /= seen
    acc_pre /= seen
    acc_rms /= seen
    acc_dt /= seen
    print("VAL  recall  {:.3f}  precision {:.3f}  RMSD {:.3f}  avg inference time  {:5.3f} ms".
          format(acc_rec, acc_pre, acc_rms, 1000*dt))


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
    validate(model, device, params, save_plot=args.plot)


if __name__ == "__main__":
    main()
