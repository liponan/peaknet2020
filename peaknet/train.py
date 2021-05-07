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
from loss import PeaknetBCELoss, PeakNetBCE1ChannelLoss
import visualize
import shutil
import argparse


def check_existence(exp, run):
    files = glob("/reg/d/psdm/cxi/{}/xtc/*{}*.xtc".format(exp, run))
    return len(files) > 0


def train(model, device, params, writer):
    model.train()
    if params["n_classes"] == 3:
        loss_func = PeaknetBCELoss(coor_scale=params["coor_scale"], pos_weight=params["pos_weight"]).to(device)
    elif params["n_classes"] == 1:
        loss_func = PeakNetBCE1ChannelLoss(coor_scale=params["coor_scale"], pos_weight=params["pos_weight"]).to(device)
    else:
        print("Unrecognized number of classes for loss function.")
        return
    train_dataset = PSANADataset(params["run_dataset_path"], subset="train", shuffle=True)
    seen = 0
    optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    print("train_dataset", len(train_dataset))

    # Preloading for visualization
    idx_experiment_visualization = 0
    cxi_path_vis, exp_vis, run_vis = train_dataset[idx_experiment_visualization]
    psana_images_vis = PSANAImage(cxi_path_vis, exp_vis, run_vis, downsample=params["downsample"], n=params["n_per_run"])

    idx_event_visualization = len(psana_images_vis) // 2
    img_vis, target_vis = psana_images_vis[idx_event_visualization]

    total_steps = 0
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
            metrics = loss_func(scores, y, verbose=params["verbose"], cutoff=params["cutoff"])
            loss = metrics["loss"]

            visualize.scalar_metrics(writer, metrics, total_steps)
            total_steps += 1
            # print("total_steps", total_steps)

            loss.backward()
            optimizer.step()
            with torch.no_grad():
                seen += n
                # print("seen {:6d}  loss {:7.5f}  recall  {:.3f}  precision {:.3f}  RMSD {:.3f}".
                #       format(seen, float(loss.data.cpu()), metrics["recall"], metrics["precision"], metrics["rmsd"]))
                print_str = "seen " + str(seen) + " ; "
                for (key, value) in metrics.items():
                    if key == "loss":
                        print_str += key + " " + str(float(value.data.cpu())) + " ; "
                    else:
                        print_str += key + " " + str(value) + " ; "
                print(print_str)
                if seen % (params["backup_every"]) == 0:
                    torch.save(model.state_dict(), "debug/"+params["experiment_name"]+"/model.pt")
                if total_steps % params["show_image_every"] == 0:
                    visualize.show_GT_prediction_image(writer, img_vis, target_vis, total_steps, params, device, model)
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
    model = UNet(n_channels=1, n_classes=params["n_classes"], n_filters=params["n_filters"])
    if args.model:
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")
    model = model.to(device)

    model_dir = os.path.join('debug', params["experiment_name"])

    if os.path.exists(model_dir):
        y = 'y'
        val = input("The model directory %s exists. Overwrite? (y/n)" % model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)
            print("Directory removed.")

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    if os.path.exists(summaries_dir):
        shutil.rmtree(summaries_dir)
    os.makedirs(summaries_dir)

    writer = SummaryWriter(summaries_dir)

    train(model, device, params, writer)
    
    
if __name__ == "__main__":
    main()
