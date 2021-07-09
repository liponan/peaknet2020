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
from models import AdaFilter_0, AdaFilter_1, AdaFilter_2
from loss import PeaknetBCELoss, PeakNetBCE1ChannelLoss
from saver import Saver
import visualize
import shutil
import argparse
import time
import numpy as np


def check_existence(exp, run):
    files = glob("/reg/d/psdm/cxi/{}/xtc/*{}*.xtc".format(exp, run))
    return len(files) > 0


def train(model, device, params, writer):
    model.train()

    print("")

    print("*** Parameters ***")
    for key, value in params.items():
        print(str(key) + ' : ' + str(value))

    print('')
    print("Will show intermediate activation: ") + str(hasattr(model, 'can_show_inter_act') and model.can_show_inter_act) + "."

    if params["n_classes"] == 1:
        loss_func = PeakNetBCE1ChannelLoss(params, device).to(device)
    else:
        print("Unrecognized number of classes for loss function.")
        return

    saver = Saver(params["saver_type"], params)

    train_dataset = PSANADataset(params["run_dataset_path"], subset="train", shuffle=False, n=params["n_experiments"])
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    # print("train_dataset", len(train_dataset))

    # Preloading for visualization
    idx_experiment_visualization = 0
    cxi_path_vis, exp_vis, run_vis = train_dataset[idx_experiment_visualization]
    psana_images_vis = PSANAImage(cxi_path_vis, exp_vis, run_vis, downsample=params["downsample"],
                                  n=params["n_per_run"], min_det_peaks=params["min_det_peaks"],
                                  use_indexed_peaks=params["use_indexed_peaks"],
                                  n_classes = params["n_classes"])
    idx_event_visualization = len(psana_images_vis) // 2
    print('')
    print('Loading image for visualization...')
    img_vis, target_vis, _ = psana_images_vis[idx_event_visualization]
    print("nPeaks visualization: " + str(len(np.nonzero(target_vis[:, 0, :, :]))))
    if params["use_indexed_peaks"]:
        print("nIndexedPeaks visualization: " + str(len(np.nonzero(target_vis[:, 1, :, :]))))


    total_steps = 0
    seen = 0
    seen_and_missed = 0
    for epoch in range(params["n_epochs"]):
        print("")
        print("*** Epoch "+str(epoch)+" ***")
        print("")
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
                                      min_det_peaks=params["min_det_peaks"], use_indexed_peaks=params["use_indexed_peaks"],
                                      n_classes = params["n_classes"])
            data_loader = DataLoader(psana_images, batch_size=params["batch_size"], shuffle=True, drop_last=True,
                                     num_workers=params["num_workers"])
            for j, (x, y, n_trials) in enumerate(data_loader):
                tic = time.time()
                optimizer.zero_grad()
                n = x.size(0)
                seen += n
                seen_and_missed += n_trials.sum().item()
                y = y.view(-1, y.size(2), y.size(3), y.size(4))
                x = x.to(device)
                y = y.to(device)

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
                        saver.upload(metrics, params["save_name"])
                    if seen % (params["backup_every"]) == 0:
                        torch.save(model.state_dict(), "debug/"+params["experiment_name"]+"/model.pt")
                    if seen % params["show_image_every"] == 0:
                        visualize.show_GT_prediction_image(writer, img_vis, target_vis, total_steps, params, device,
                                                           model, use_indexed_peaks=params["use_indexed_peaks"])
                        # visualize.show_weights_model(writer, model, total_steps)
                        if hasattr(model, 'can_show_inter_act') and model.can_show_inter_act:
                            visualize.show_inter_act(writer, img_vis, total_steps, params, device, model)
            psana_images.close()
    saver.save(params["save_name"])
    torch.save(model, "debug/"+params["experiment_name"]+"/model.pt")
    print("Model saved at " + "debug/"+params["experiment_name"]+"/model.pt.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("params", type=str, default=None, help="A json file")

    # System parameters
    p.add_argument("--gpu", "-g", type=int, default=0, help="Use GPU x")

    # Existing model
    p.add_argument("--model", "-m", type=str, default=None, help="A .PT file")

    # Parameters not in params.json (can be easily modified when calling train.py)
    p.add_argument("--experiment_name", type=str, default=None)
    p.add_argument("--run_dataset_path", type=str, default="/cds/home/a/axlevy/peaknet2020/data/cxic0415_psocake2.csv")
    p.add_argument("--n_classes", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=0.)
    p.add_argument("--pos_weight", type=float, default=1e-1)
    p.add_argument("--gamma", type=float, default=1.)
    p.add_argument("--use_focal_loss", type=str, default="False")
    p.add_argument("--gamma_FL", type=float, default=1.)
    p.add_argument("--cutoff", type=float, default=0.5)
    p.add_argument("--n_experiments", type=int, default=-1)
    p.add_argument("--n_per_run", type=int, default=-1)
    p.add_argument('--confirm_delete', dest='confirm_delete', action='store_true')
    p.add_argument('--no_confirm_delete', dest='confirm_delete', action='store_false')
    p.set_defaults(confirm_delete=False)
    p.add_argument("--saver_type", type=str, default=None)
    p.add_argument("--save_name", type=str, default=None)
    p.add_argument("--backup_every", type=int, default=500)
    p.add_argument("--print_every", type=int, default=25)
    p.add_argument("--show_image_every", type=int, default=100)
    p.add_argument("--upload_every", type=int, default=100)
    p.add_argument("--min_det_peaks", type=int, default=100)
    p.add_argument("--n_epochs", type=int, default=3)
    p.add_argument("--use_indexed_peaks", type=str, default="True")
    p.add_argument("--downsample", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--use_adaptive_filtering", type=str, default="True")
    p.add_argument("--use_scheduled_pos_weight", type=str, default="False")
    p.add_argument("--pos_weight_0", type=float, default=1e2)
    p.add_argument("--annihilation_speed", type=float, default=1e-1)
    p.add_argument("--step_after", type=int, default=200)
    return p.parse_args()

def load_model(params):
    if params["model"] == "UNet":
        model = UNet(n_channels=1, n_classes=params["n_classes"], n_filters=params["n_filters"], params=params)
    elif params["model"] == "model_0":
        model = AdaFilter_0(params=params)
    elif params["model"] == "model_1":
        model = AdaFilter_1(params=params)
    elif params["model"] == "model_2":
        model = AdaFilter_2(params=params)
    else:
        print("Unrecognized model.")
        model = None
    return model


def main():
    args = parse_args()
    params = json.load(open(args.params))

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
    params["run_dataset_path"] = args.run_dataset_path
    params["n_classes"] = args.n_classes
    params["lr"] = args.lr
    params["weight_decay"] = args.weight_decay
    params["pos_weight"] = args.pos_weight
    params["gamma"] = args.gamma
    params["gamma_FL"] = args.gamma_FL
    params["cutoff"] = args.cutoff
    params["n_experiments"] = args.n_experiments
    params["n_per_run"] = args.n_per_run
    params["saver_type"] = args.saver_type
    params["save_name"] = args.save_name
    params["backup_every"] = args.backup_every
    params["print_every"] = args.print_every
    params["show_image_every"] = args.show_image_every
    params["upload_every"] = args.upload_every
    params["min_det_peaks"] = args.min_det_peaks
    params["downsample"] = args.downsample
    params["num_workers"] = args.num_workers
    params["n_epochs"] = args.n_epochs
    params["pos_weight_0"] = args.pos_weight_0
    params["annihilation_speed"] = args.annihilation_speed
    params["step_after"] = args.step_after
    if args.use_indexed_peaks == "True":
        params["use_indexed_peaks"] = True
    else:
        params["use_indexed_peaks"] = False
    if args.use_adaptive_filtering == "True":
        params["use_adaptive_filtering"] = True
    else:
        params["use_adaptive_filtering"] = False
    if args.use_focal_loss == "True":
        params["use_focal_loss"] = True
    else:
        params["use_focal_loss"] = False
    if args.use_scheduled_pos_weight == "True":
        params["use_scheduled_pos_weight"] = True
        params["step_after"] = 2000 / params["batch_size"] # hard encoded
        if params["use_focal_loss"]:
            params["pos_weight_0"] = 100 # hard encoded
        else:
            params["pos_weight_0"] = 1 # hard encoded
    else:
        params["use_scheduled_pos_weight"] = False
    params["verbose"] = False

    model = load_model(params)

    # Existing model
    if args.model:
        model.load_state_dict(torch.load(args.model, map_location="cpu"))

    model = model.to(device)

    model_dir = os.path.join('debug', params["experiment_name"])

    if os.path.exists(model_dir):
        y = 'y'
        if args.confirm_delete:
            print('')
            val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
        else:
            val = 'y'

        if val == 'y':
            shutil.rmtree(model_dir)
            print('')
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
