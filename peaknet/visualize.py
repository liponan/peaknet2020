import numpy as np
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
        writer.add_image(panel_name, fig, global_step=total_steps)