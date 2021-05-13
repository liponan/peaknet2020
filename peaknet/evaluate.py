import os
from glob import glob
import json
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from data import PSANADataset, PSANAImage
from unet import UNet
from saver import Saver
import shutil
import argparse

def evaluation_metrics(scores, y, cutoff=0.5):
    scores_c = scores[:, 0, :, :].reshape(-1)
    targets_c = y[:, 0, :, :].reshape(-1)
    gt_mask = targets_c > 0

    n_gt = targets_c.sum()
    positives = (nn.Sigmoid()(scores_c) > cutoff)
    n_p = positives.sum()
    n_tp = (positives[gt_mask]).sum()
    if n_p == 0:
        n_p = n_tp
    recall = float(n_tp) / max(1, int(n_gt))
    precision = float(n_tp) / max(1, int(n_p))

    metrics = {"recall": recall, "precision": precision}
    return metrics

def check_existence(exp, run):
    files = glob("/reg/d/psdm/cxi/{}/xtc/*{}*.xtc".format(exp, run))
    return len(files) > 0

def evaluate(model, device, params):
    model.eval()

    saver = Saver(params["saver_type"], params)

    eval_dataset = PSANADataset(model.dataset_path, subset="val", shuffle=True, n=params["n_experiments"])
    seen = 0

    total_steps = 0
    with torch.no_grad():
        for i, (cxi_path, exp, run) in enumerate(eval_dataset):
            if check_existence(exp, run):
                pass
            else:
                print("[{:}] exp: {}  run: {}  PRECHECK FAILED".format(i, exp, run))
                continue
            print("*********************************************************************")
            print("[{:}] exp: {}  run: {}\ncxi: {}".format(i, exp, run, cxi_path))
            print("*********************************************************************")
            psana_images = PSANAImage(cxi_path, exp, run, downsample=model.downsample, n=params["n_per_run"])
            data_loader = DataLoader(psana_images, batch_size=1, shuffle=True, drop_last=True,
                                     num_workers=0)
            for j, (x, y) in enumerate(data_loader):
                n = x.size(0)
                h, w = x.size(2), x.size(3)
                x = x.view(-1, 1, h, w).to(device)  # each panel is treated independently !!0
                y = y.view(-1, 3, h, w).to(device)
                scores = model(x)
                metrics = evaluation_metrics(scores, y, cutoff=params["cutoff_eval"])

                total_steps += 1
                seen += n

                if total_steps % params["print_every"] == 0:
                    print_str = "seen " + str(seen) + " ; "
                    for (key, value) in metrics.items():
                        print_str += key + " " + str(value) + " ; "
                    print(print_str)
                if total_steps % params["upload_every"] == 0:
                    saver.upload(metrics)
            psana_images.close()
        saver.save(params["save_name"])

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)

    # Existing model
    p.add_argument("--model_path", "-m", required=True, type=str, default=None, help="A path to .PT file")

    # System parameters
    p.add_argument("--gpu", "-g", type=int, default=None, help="Use GPU x")

    # Parameters that can be modified when calling evaluate.py
    p.add_argument("--cutoff_eval", type=float, default=0.5)
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument("--upload_every", type=int, default=1)
    p.add_argument("--saver_type", type=str, default="precision_recall_evaluation")
    p.add_argument("--save_name", type=str, default=None)
    p.add_argument("--n_experiments", type=int, default=-1)
    p.add_argument("--n_per_run", type=int, default=50000)

    return p.parse_args()


def main():
    args = parse_args()

    # Existing model
    model = torch.load(args.model_path)

    # System parameters
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")

    model = model.to(device)

    params = {}
    params["cutoff_eval"] = args.cutoff_eval
    params["print_every"] = args.print_every
    params["upload_every"] = args.upload_every
    params["saver_type"] = args.saver_type
    params["save_name"] = args.save_name
    params["n_experiments"] = args.n_experiments
    params["n_per_run"] = args.n_per_run

    evaluate(model, device, params)


if __name__ == "__main__":
    main()