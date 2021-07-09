import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn as nn

def scalar_metrics(writer, metrics, total_steps):
    for (key, value) in metrics.items():
        writer.add_scalar(key, value, total_steps)

def show_weights_model(writer, model, total_steps):
    print("*** WEIGHTS ***")
    print(model.state_dict()['ada_filter.0.weight'])
    for key, item in enumerate(model.state_dict()):
        print(key)
        print(item.shape)
    # gen_peak_finding_w = model.state_dict()['gen_peak_finding.0.1.weight'][:, 0].cpu().numpy()
    # channels = gen_peak_finding_w.shape[0]
    # fig, axs = plt.subplots(2, 4, figsize=(10, 5))
    # for c in range(min(8, channels)):
    #     axs[c // 4, c % 4].imshow(gen_peak_finding_w[c], cmap='gray')
    # writer.add_figure('Generic Peak Finding Weights (0.1)', fig, global_step=total_steps)

def show_GT_prediction_image(writer, img_vis, target_vis, total_steps, params, device, model, use_indexed_peaks=False):
    print("*** PANELS ***")

    center_panels = [0, 1, 8, 9, 16, 17, 24, 25]
    top_left = [4, 5]

    h, w = img_vis.size(1), img_vis.size(2)
    x = img_vis.view(1, -1, h, w).to(device)
    scores = model(x)
    scores = nn.Sigmoid()(scores).cpu().numpy()
    if model.downsample_bool:
        img_vis = model.downsample_for_visualization(x).cpu().numpy()
        img_vis = img_vis.reshape(-1, img_vis.shape[2], img_vis.shape[3])

    for i in center_panels + top_left:
        panel_name = 'panel_'+str(i)

        fig = plt.figure(figsize=(10,10))
        plt.subplot(211)
        plt.imshow(img_vis[i], cmap='Blues')
        plt.xticks([])
        plt.yticks([])

        # Prediction
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
                     'rs', markerfacecolor='none', markersize=5, markeredgewidth=2.0, alpha=.8, label="model")
        else:
            print("Unrecognized number of classes for visualization.")

        # GT peak finding
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
                     'gs', markerfacecolor='none', markersize=10, markeredgewidth=2.0, alpha=.8, label="psocake")
        else:
            print("Unrecognized number of classes for visualization.")

        # GT indexing
        if use_indexed_peaks:
            indices_nonzero = np.array(np.nonzero(target_vis[i, 1]))
            if params["n_classes"] == 1:
                plt.plot(indices_nonzero[:, 1] - .5,
                         indices_nonzero[:, 0] - .5,
                         'mo', markerfacecolor='none', markersize=10, markeredgewidth=2.0, alpha=.8, label="indexing")
            else:
                print("Unrecognized number of classes for visualization.")
        plt.legend(loc="best")

        plt.subplot(212)
        plt.imshow(img_vis[i], cmap='Blues')
        plt.xticks([])
        plt.yticks([])

        writer.add_figure(panel_name, fig, global_step=total_steps)

def show_inter_act(writer, img_vis, total_steps, params, device, model):
    print("*** INTERMEDIATE ACTIVATIONS ***")

    panels = [0, 4]

    h, w = img_vis.size(1), img_vis.size(2)
    x = img_vis.view(1, -1, h, w).to(device)
    x_ds, filtered_x, logits, logits_out = model(x, return_intermediate_act=True)
    x_ds = x_ds.cpu().numpy().reshape(-1, 1, x_ds.size(2), x_ds.size(3))
    filtered_x = filtered_x.cpu().numpy()
    logits = logits.cpu().numpy()
    logits_out = logits_out.cpu().numpy()

    for i in panels:
        title = 'Intermediate Activations Panel ' + str(i)

        fig = plt.figure(figsize=(10, 20))

        plt.subplot(511)
        plt.title("Original Panel")
        plt.imshow(img_vis[i], cmap='Blues')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])

        plt.subplot(512)
        plt.title("Downsampled Panel")
        plt.imshow(x_ds[i, 0], cmap='Blues')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])

        plt.subplot(513)
        plt.title("After Panel/Exp-Dependent Filtering")
        plt.imshow(filtered_x[i, 0], cmap='Blues')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])

        plt.subplot(514)
        plt.title("After Generic Peak Finding")
        plt.imshow(logits[i, 0], cmap='Blues')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])

        plt.subplot(515)
        plt.title("After Panel-Dependent Scaling")
        plt.imshow(logits_out[i, 0], cmap='Blues')
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])

        writer.add_figure(title, fig, global_step=total_steps)