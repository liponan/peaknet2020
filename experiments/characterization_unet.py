# Influence of pos_weight

import numpy as np
import os

n_experiments = 1
pos_weight_list = np.logspace(-2, 0, 5)

for i, pw in enumerate(pos_weight_list):
    print("Experiment #"+str(i + 1))
    print("pos_weight: "+str(pw))
    os.system('python ../peaknet/train.py params.json -g 0 --pos_weight '+str(pw))