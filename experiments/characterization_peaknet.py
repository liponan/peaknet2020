import numpy as np
import os

os.chdir("/cds/home/a/axlevy/peaknet2020/peaknet")

index_experiment = 1

# Experiment #1: peaknet and unet vs pos_weight
if index_experiment == 1:
    pos_weight_list = [1, 1e-1, 1e-2]

    offset_idx = 0
    prefix = "pos_weight_peaknet_"
    for i, pw in enumerate(pos_weight_list):
        print("---")
        print("Experiment #" + str(i + 1))
        print("pos_weight: " + str(pw))
        print("---")
        save_name = prefix + str(offset_idx + i)
        os.system('python train.py params_model_1.json --saver_type "precision_recall"'
                  ' --n_experiments -1 --n_per_run -1 --n_epochs 1 --show_image_every 10000'
                  ' --save_name ' + str(save_name) + ' --pos_weight ' + str(pw))

    offset_idx = 0
    prefix = "pos_weight_unet_"
    for i, pw in enumerate(pos_weight_list):
        print("---")
        print("Experiment #" + str(i + 1))
        print("pos_weight: " + str(pw))
        print("---")
        save_name = prefix + str(offset_idx + i)
        os.system('python train.py params.json --saver_type "precision_recall"'
                  ' --n_experiments -1 --n_per_run -1 --n_epochs 1 --show_image_every 10000'
                  ' --save_name ' + str(save_name) + ' --pos_weight ' + str(
            pw))