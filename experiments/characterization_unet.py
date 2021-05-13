# Influence of pos_weight

import numpy as np
import os

pos_weight_list = np.logspace(-2, 0, 5)

os.chdir("/cds/home/a/axlevy/peaknet2020/peaknet")

for i, pw in enumerate(pos_weight_list):
    print("---")
    print("Experiment #"+str(i + 1))
    print("pos_weight: "+str(pw))
    print("---")
    save_name = "pos_weight_"+str(i)
    os.system('python train.py params.json -g 0 --no_confirm_delete --n_experiments 1 --n_per_run 50'
              ' --saver_type="precision_recall" --save_name='+str(save_name)+'  --pos_weight='+str(pw))