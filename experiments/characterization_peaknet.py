import numpy as np
import os

os.chdir("/cds/home/a/axlevy/peaknet2020/peaknet")

index_experiment = 5

print("Index Experiment: " + str(index_experiment))

# Experiment #1: peaknet and unet vs pos_weight
if index_experiment == 1:
    pos_weight_list = [1e-3, 1e-4]

    offset_idx = 3
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

    pos_weight_list = [1e-2, 1e-3, 1e-4]
    offset_idx = 2
    prefix = "pos_weight_unet_"
    for i, pw in enumerate(pos_weight_list):
        print("---")
        print("Experiment #" + str(i + 1))
        print("pos_weight: " + str(pw))
        print("---")
        save_name = prefix + str(offset_idx + i)
        os.system('python train.py params.json --saver_type "precision_recall"'
                  ' --n_experiments -1 --n_per_run -1 --n_epochs 1 --show_image_every 10000'
                  ' --save_name ' + str(save_name) + ' --pos_weight ' + str(pw))

# Experiment #2: influence of indexing on unet
if index_experiment == 2:
    pos_weight_list = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    offset_idx = 0
    prefix = "pos_weight_no_idxg_unet_"
    for i, pw in enumerate(pos_weight_list):
        print("---")
        print("Experiment #" + str(i + 1))
        print("pos_weight: " + str(pw))
        print("---")
        save_name = prefix + str(offset_idx + i)
        os.system('python train.py params.json --saver_type "precision_recall"'
                  ' --n_experiments -1 --n_per_run -1 --n_epochs 1 --show_image_every 10000'
                  ' --use_indexed_peaks False --save_name ' + str(save_name) + ' --pos_weight ' + str(pw))

# Experiment #3: influence of indexing on PeakNet 2.0
if index_experiment == 3:
    pos_weight_list = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    offset_idx = 0
    prefix = "pos_weight_no_idxg_peaknet_"
    for i, pw in enumerate(pos_weight_list):
        print("---")
        print("Experiment #" + str(i + 1))
        print("pos_weight: " + str(pw))
        print("---")
        save_name = prefix + str(offset_idx + i)
        os.system('python train.py params_model_1.json --saver_type "precision_recall"'
                  ' --n_experiments -1 --n_per_run -1 --n_epochs 1 --show_image_every 10000'
                  ' --use_indexed_peaks False --save_name ' + str(save_name) + ' --pos_weight ' + str(pw))

# Experiment #4: influence of adaptive filtering on PeakNet 2.0
if index_experiment == 4:
    pos_weight_list = [1, 1e-1, 1e-2, 1e-3, 1e-4]
    offset_idx = 0
    prefix = "pos_weight_no_ada_filt_peaknet_"
    for i, pw in enumerate(pos_weight_list):
        print("---")
        print("Experiment #" + str(i + 1))
        print("pos_weight: " + str(pw))
        print("---")
        save_name = prefix + str(offset_idx + i)
        os.system('python train.py params_model_1.json --saver_type "precision_recall"'
                  ' --n_experiments -1 --n_per_run -1 --n_epochs 1 --show_image_every 10000'
                  ' --use_indexed_peaks True --use_adaptive_filtering False'
                  ' --save_name ' + str(save_name) + ' --pos_weight ' + str(pw))

# Experiment #5: BCE vs FL on PeakNet with indexing and AF
if index_experiment == 5:
    pos_weight_list = [1e3, 1e2, 1e1]
    offset_idx = 0
    prefix = "pos_weight_FL_peaknet_"
    for i, pw in enumerate(pos_weight_list):
        print("---")
        print("Experiment #" + str(i + 1))
        print("pos_weight: " + str(pw))
        print("---")
        save_name = prefix + str(offset_idx + i)
        os.system('python train.py params_model_1.json --saver_type "precision_recall"'
                  ' --n_experiments 5 --n_per_run -1 --n_epochs 1 --show_image_every 10000'
                  ' --use_indexed_peaks True --use_adaptive_filtering True --use_focal_loss True'
                  ' --save_name ' + str(save_name) + ' --pos_weight ' + str(pw))