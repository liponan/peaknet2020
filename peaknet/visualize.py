import numpy as np
import os
import matplotlib.pyplot as plt

def scalar_metrics(writer, metrics, total_steps):
    for (key, value) in metrics.items():
        writer.add_scalar(key, value, total_steps)

def show_GT_image(writer, img_vis, target_vis, total_steps, n=5):
    for i in range(n):
        panel_name = 'panel_'+str(i)
        # writer.add_image(panel_name, img_vis[i][None, :, :], global_step=total_steps)
        fig = plt.figure()
        plt.imshow(img_vis[i])
        plt.xticks([])
        plt.yticks([])
        indices_nonzero = np.array(np.nonzero(target_vis[i, 0]))
        shift_u = target_vis[i, 1, indices_nonzero[:, 0], indices_nonzero[:, 1]].numpy()
        shift_v = target_vis[i, 2, indices_nonzero[:, 0], indices_nonzero[:, 1]].numpy()
        print(shift_u)
        plt.plot(indices_nonzero[:, 0] - .5 + shift_u,
                 indices_nonzero[:, 1] - .5 + shift_v,
                 'rs', markerfacecolor='none', markersize=15, markeredgewidth=2.0) # not sure of orientation here...
        writer.add_figure(panel_name, fig, global_step=total_steps)