# Influence of pos_weight

import numpy as np
import os

os.chdir("/cds/home/a/axlevy/peaknet2020/peaknet")

index_experiment = 2

# Experiment #1
if index_experiment == 1:
    pos_weight_list = np.logspace(-4, 0, 5)
    for i, pw in enumerate(pos_weight_list):
        print("---")
        print("Experiment #" + str(i + 1))
        print("pos_weight: " + str(pw))
        print("---")
        save_name = "pos_weight_" + str(i)
        os.system('python train.py params.json -g 0 --no_confirm_delete --n_experiments -1 --n_per_run 50000'
                  ' --saver_type "precision_recall" --save_name ' + str(save_name) + ' --pos_weight ' + str(pw))

# Experiment #2
if index_experiment == 2:
    experiment_name = "unet_pw1em3"
    training_required = False
    pw = 1e-3
    save_prefix = "eval_cutoff_"

    if training_required:
        print("---")
        print("Training Phase")
        print("---")
        os.system('python train.py params.json -g 0 --no_confirm_delete '
                  '--experiment_name ' + experiment_name + ' --pos_weight=' + str(pw))
        print("")
    print("---")
    print("Evaluation Phase")
    print("---")
    model_path = "debug/" + experiment_name + "/model.pt"
    cutoff_eval_list = [1e-3, 1e-2, 5e-2, 1e-1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999]
    cutoff_eval_list = cutoff_eval_list[9:]
    offset_idx = 9
    for i, cutoff_eval in enumerate(cutoff_eval_list):
        save_name = save_prefix + str(offset_idx + i)
        print("---")
        print("Experiment #" + str(i + 1))
        print("cutoff_eval: " + str(cutoff_eval))
        print("save_name: " + str(save_name))
        print("---")
        os.system('python evaluate.py --model_path ' + model_path + ' -g 0 '
                  '--cutoff_eval ' + str(cutoff_eval) + ' --saver_type "precision_recall_evaluation" '
                  '--save_name ' + str(save_name))

# Experiment #3
if index_experiment == 3:
    experiment_name = "unet_pw1em4"
    training_required = True
    pw = 1e-4
    save_prefix = "eval_cutoff_2_"

    if training_required:
        print("---")
        print("Training Phase")
        print("---")
        os.system('python train.py params.json -g 0 --no_confirm_delete '
                  '--experiment_name ' + experiment_name + ' --pos_weight=' + str(pw))
        print("")
    print("---")
    print("Evaluation Phase")
    print("---")
    model_path = "debug/" + experiment_name + "/model.pt"
    cutoff_eval_list = [1e-3, 1e-2, 5e-2, 1e-1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
    offset_idx = 0
    for i, cutoff_eval in enumerate(cutoff_eval_list):
        save_name = save_prefix + str(offset_idx + i)
        print("---")
        print("Experiment #" + str(i + 1))
        print("cutoff_eval: " + str(cutoff_eval))
        print("save_name: " + str(save_name))
        print("---")
        os.system('python evaluate.py --model_path ' + model_path + ' -g 0 '
                  '--cutoff_eval ' + str(cutoff_eval) + ' --saver_type "precision_recall_evaluation" '
                  '--save_name ' + str(save_name))