import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn

def scalar_metrics(writer, metrics, total_steps):
    for (key, value) in metrics.items():
        writer.add_scalar(key, value, total_steps)

def show_GT_prediction_image(writer, img_vis, target_vis, total_steps, params, device, model, n=32):
    for i in range(n):
        panel_name = 'panel_'+str(i)

        fig = plt.figure(figsize=(10,10))
        plt.subplot(211)
        plt.imshow(img_vis[i], cmap='Blues')
        plt.xticks([])
        plt.yticks([])

        # Prediction
        h, w = x.size(1), x.size(2)
        x = img_vis.view(1, -1, h, w).to(device)
        scores = model(x)
        scores = nn.Sigmoid()(scores).cpu().numpy()
        indices_nonzero = np.array(np.argwhere(scores[i, 0] > params["cutoff"]))
        if params["n_classes"] == 3:
            shift_u = scores[i, 1, indices_nonzero[:, 0], indices_nonzero[:, 1]].numpy()
            shift_v = scores[i, 2, indices_nonzero[:, 0], indices_nonzero[:, 1]].numpy()
            plt.plot(indices_nonzero[:, 1] - .5 + shift_v,
                     indices_nonzero[:, 0] - .5 + shift_u,
                     'rs', markerfacecolor='none', markersize=5, markeredgewidth=2.0, alpha=.8)
        elif params["n_classes"] == 1:
            plt.plot(indices_nonzero[:, 1] - .5,
                     indices_nonzero[:, 0] - .5,
                     'rs', markerfacecolor='none', markersize=5, markeredgewidth=2.0, alpha=.8)
        else:
            print("Unrecognized number of classes for visualization.")

        # GT
        indices_nonzero = np.array(np.nonzero(target_vis[i, 0]))
        if params["n_classes"] == 3:
            shift_u = target_vis[i, 1, indices_nonzero[:, 0], indices_nonzero[:, 1]].numpy()
            shift_v = target_vis[i, 2, indices_nonzero[:, 0], indices_nonzero[:, 1]].numpy()
            plt.plot(indices_nonzero[:, 1] - .5 + shift_v,
                     indices_nonzero[:, 0] - .5 + shift_u,
                     'gs', markerfacecolor='none', markersize=10, markeredgewidth=2.0, alpha=.8)
        elif params["n_classes"] == 1:
            plt.plot(indices_nonzero[:, 1] - .5,
                     indices_nonzero[:, 0] - .5,
                     'gs', markerfacecolor='none', markersize=10, markeredgewidth=2.0, alpha=.8)
        else:
            print("Unrecognized number of classes for visualization.")

        plt.subplot(212)
        plt.imshow(img_vis[i], cmap='Blues')
        plt.xticks([])
        plt.yticks([])

        writer.add_figure(panel_name, fig, global_step=total_steps)