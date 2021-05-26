import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('No display found. Using non-interactive Agg backend.')
    mpl.use('Agg')
from glob import glob
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import PSANADataset, PSANAImage
from unet import UNet
from models import AdaFilter_0
from loss import PeaknetBCELoss, PeakNetBCE1ChannelLoss
from saver import Saver
import visualize
import shutil
import argparse
import time


def check_existence(exp, run):
    files = glob("/reg/d/psdm/cxi/{}/xtc/*{}*.xtc".format(exp, run))
    return len(files) > 0


def train(model, device, params, writer):
    model.train()

    if params["n_classes"] == 3:
        loss_func = PeaknetBCELoss(coor_scale=params["coor_scale"], pos_weight=params["pos_weight"], device=device).to(device)
    elif params["n_classes"] == 1:
        loss_func = PeakNetBCE1ChannelLoss(pos_weight=params["pos_weight"], device=device).to(device)
    else:
        print("Unrecognized number of classes for loss function.")
        return

    saver = Saver(params["saver_type"], params)

    train_dataset = PSANADataset(params["run_dataset_path"], subset="train", shuffle=True, n=params["n_experiments"])
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    # print("train_dataset", len(train_dataset))

    # Preloading for visualization
    idx_experiment_visualization = 0
    cxi_path_vis, exp_vis, run_vis = train_dataset[idx_experiment_visualization]
    psana_images_vis = PSANAImage(cxi_path_vis, exp_vis, run_vis, downsample=params["downsample"], n=params["n_per_run"])

    idx_event_visualization = len(psana_images_vis) // 2
    print("idx_event_visualization: "+str(idx_event_visualization))
    img_vis, target_vis = psana_images_vis[idx_event_visualization]

    total_steps = 0
    seen = 0
    seen_and_missed = 0
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
        psana_images = PSANAImage(cxi_path, exp, run, downsample=params["downsample"], n=params["n_per_run"],
                                  min_det_peaks=params["min_det_peaks"])
        data_loader = DataLoader(psana_images, batch_size=params["batch_size"], shuffle=True, drop_last=True,
                                 num_workers=params["num_workers"])
        for j, (x, y, n_trials) in enumerate(data_loader):
            tic = time.time()
            optimizer.zero_grad()
            n = x.size(0)
            seen += n
            seen_and_missed += n_trials.sum()
            h, w = x.size(2), x.size(3)
            x = x.to(device)
            y = y.to(device)
            y = y.view(-1, 3, h, w).to(device)

            scores = model(x)
            metrics = loss_func(scores, y, verbose=params["verbose"], cutoff=params["cutoff"])
            loss = metrics["loss"]

            visualize.scalar_metrics(writer, metrics, total_steps)
            total_steps += 1

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                if seen % params["print_every"] == 0:
                    toc = time.time()
                    print(str((toc - tic) / params["batch_size"] * 1e3) + " ms per sample")
                    print_str = "seen " + str(seen) + " ; "
                    ratio_real_hits = seen / seen_and_missed
                    print_str += "ratio used " + str(ratio_real_hits) + " ; "
                    for (key, value) in metrics.items():
                        if key == "loss":
                            print_str += key + " " + str(float(value.data.cpu())) + " ; "
                        else:
                            print_str += key + " " + str(value) + " ; "
                    print(print_str)
                if seen % params["upload_every"] == 0:
                    saver.upload(metrics)
                if seen % (params["backup_every"]) == 0:
                    torch.save(model.state_dict(), "debug/"+params["experiment_name"]+"/model.pt")
                if seen % params["show_image_every"] == 0:
                    visualize.show_GT_prediction_image(writer, img_vis, target_vis, total_steps, params, device, model)
        psana_images.close()
    saver.save(params["save_name"])
    torch.save(model, "debug/"+params["experiment_name"]+"/model.pt")
    print("Model saved at " + "debug/"+params["experiment_name"]+"/model.pt.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("params", type=str, default=None, help="A json file")

    # System parameters
    p.add_argument("--gpu", "-g", type=int, default=None, help="Use GPU x")

    # Existing model
    p.add_argument("--model", "-m", type=str, default=None, help="A .PT file")

    # Parameters not in params.json (can be easily modified when calling train.py)
    p.add_argument("--experiment_name", type=str, default=None)
    p.add_argument("--pos_weight", type=float, default=1e-2)
    p.add_argument("--cutoff", type=float, default=0.5)
    p.add_argument("--n_experiments", type=int, default=-1)
    p.add_argument("--n_per_run", type=int, default=50000)
    p.add_argument('--confirm_delete', dest='confirm_delete', action='store_true')
    p.add_argument('--no_confirm_delete', dest='confirm_delete', action='store_false')
    p.set_defaults(confirm_delete=True)
    p.add_argument("--saver_type", type=str, default=None)
    p.add_argument("--save_name", type=str, default=None)
    p.add_argument("--backup_every", type=int, default=500)
    p.add_argument("--print_every", type=int, default=25)
    p.add_argument("--upload_every", type=int, default=10)
    p.add_argument("--min_det_peaks", type=int, default=-1)
    return p.parse_args()

def load_model(params):
    if params["model"] == "UNet":
        model = UNet(n_channels=1, n_classes=params["n_classes"], n_filters=params["n_filters"], params=params)
    elif params["model"] == "model_0":
        model = AdaFilter_0(params=params)
    else:
        print("Unrecognized model.")
        model = None
    return model


def main():
    args = parse_args()
    params = json.load(open(args.params))
    model = load_model(params)

    # Existing model
    if args.model:
        model.load_state_dict(torch.load(args.model, map_location="cpu"))

    # System parameters
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")

    # Parameters not in params.json
    if args.experiment_name is not None:
        params["experiment_name"] = args.experiment_name
    else:
        params["experiment_name"] = params["model"]
    params["pos_weight"] = args.pos_weight
    params["cutoff"] = args.cutoff
    params["n_experiments"] = args.n_experiments
    params["n_per_run"] = args.n_per_run
    params["saver_type"] = args.saver_type
    params["save_name"] = args.save_name
    params["backup_every"] = args.backup_every
    params["print_every"] = args.print_every
    params["upload_every"] = args.upload_every
    params["min_det_peaks"] = args.min_det_peaks


    model = model.to(device)

    model_dir = os.path.join('debug', params["experiment_name"])

    if os.path.exists(model_dir):
        y = 'y'
        if args.confirm_delete:
            val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
        else:
            val = 'y'

        if val == 'y':
            shutil.rmtree(model_dir)
            print(params["experiment_name"] + " directory removed.")

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    if os.path.exists(summaries_dir):
        shutil.rmtree(summaries_dir)
    os.makedirs(summaries_dir)

    writer = SummaryWriter(summaries_dir)

    train(model, device, params, writer)
    
    
if __name__ == "__main__":
    main()
