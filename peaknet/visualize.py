import numpy as np
import matplotlib.pyplot as plt

def scalar_metrics(writer, metrics, total_steps):
    for (key, value) in metrics.items():
        writer.add_scalar(key, value, total_steps)

def show_GT_image(writer, img_vis, target_vis, total_steps):
    n = img_vis.shape[0]
    for i in range(n):
        panel_name = 'panel'+str(i)
        writer.add_image(panel_name, img_vis[i][None, :, :], global_step=total_steps)