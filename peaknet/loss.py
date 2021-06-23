import torch
import torch.nn as nn
import numpy as np

class PeaknetBCELoss(nn.Module):

    def __init__(self, coor_scale=1, pos_weight=1.0, device=None):
        super(PeaknetBCELoss, self).__init__()
        self.coor_scale = coor_scale
        self.mseloss = nn.MSELoss()
        self.bceloss = None
        self.pos_weight = torch.Tensor([pos_weight])
        if device is not None:
            self.pos_weight.to(device)

    def forward(self, scores, targets, cutoff=0.1, verbose=False):
        if verbose:
            print("scores", scores.size())
            print("targets", targets.size())
        scores_c = scores[:, 0, :, :].reshape(-1)
        targets_c = targets[:, 0, :, :].reshape(-1)
        gt_mask = targets_c > 0
        scores_x = nn.Sigmoid()(scores[:, 2, :, :].reshape(-1)[gt_mask])
        scores_y = nn.Sigmoid()(scores[:, 1, :, :].reshape(-1)[gt_mask])
        targets_x = targets[:, 2, :, :].reshape(-1)[gt_mask]
        targets_y = targets[:, 1, :, :].reshape(-1)[gt_mask]

        pos_weight = self.pos_weight * (~gt_mask).sum().double() / gt_mask.sum().double() # p_c = negative_GT / positive_GT

        self.bceloss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss_conf = self.bceloss(scores_c, targets_c)
        # loss_x = self.coor_scale * self.mseloss(scores_x, targets_x)
        # loss_y = self.coor_scale * self.mseloss(scores_y, targets_y)
        loss_x = self.mseloss(scores_x, targets_x)
        loss_y = self.mseloss(scores_y, targets_y)
        loss_coor = loss_x + loss_y
        loss = loss_conf + self.coor_scale * loss_coor # why re-multiply by coor_scale?
        with torch.no_grad():
            n_gt = targets_c.sum()
            positives = (scores_c > cutoff)
            n_p = positives.sum()
            n_tp = (positives[gt_mask]).sum()
            recall = float(n_tp) / max(1, int(n_gt))
            precision = float(n_tp) / max(1, int(n_p))
            rmsd = torch.sqrt(torch.mean((targets_x - scores_x).pow(2) + (targets_y - scores_y).pow(2)))
            if verbose:
                print("nGT", int(n_gt), "recall", int(n_tp), "nP", int(n_p), "rmsd", float(rmsd),
                      "loss", float(loss.data), "conf", float(loss_conf.data), "coor", float(loss_coor.data))
        metrics = {"loss": loss, "loss_conf": loss_conf, "loss_coor": loss_coor, "recall": recall,
                   "precision": precision, "rmsd": rmsd}
        return metrics

def focal_loss(scores, target, TN, alpha, gamma):
    n_p = TN.sum().double() / target.sum().double() # different from BCE
    scores_sigmoid = torch.clamp(nn.Sigmoid()(scores), min=1e-3, max=1. - 1e-3)
    loss = -(alpha * (target * (1. - scores_sigmoid) ** gamma * torch.log(scores_sigmoid)).mean() +
             ((1. - target) * scores_sigmoid ** gamma * torch.log(1. - scores_sigmoid)).mean()) * n_p
    return loss

def update_schedule_pos_weight(pos_weight, pos_weight_inf, annihilation_speed):
    pos_weight = (pos_weight - pos_weight_inf) * (1. - annihilation_speed) + pos_weight_inf
    print('pos_weight: ' + str(pos_weight.item()))
    return pos_weight

class PeakNetBCE1ChannelLoss(nn.Module):

    def __init__(self, params, device):
        super(PeakNetBCE1ChannelLoss, self).__init__()
        pos_weight = params["pos_weight"]
        use_indexed_peaks = params["use_indexed_peaks"]
        gamma = params["gamma"]
        use_focal_loss = params["use_focal_loss"]
        gamma_FL = params["gamma_FL"]
        use_scheduled_pos_weight = params["use_scheduled_pos_weight"]
        pos_weight_0 = params["pos_weight_0"]
        annihilation_speed = params["annihilation_speed"]
        self.use_indexed_peaks = use_indexed_peaks
        if use_indexed_peaks:
            self.maxpool_idxg = nn.Sequential(nn.ReflectionPad2d(2),
                                              nn.MaxPool2d(5, stride=1, padding=0))
        self.bceloss = None
        self.maxpool = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.MaxPool2d(3, stride=1, padding=0))
        self.pos_weight = torch.Tensor([pos_weight])
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            print('')
            print("Will use focal loss.")
            self.gamma_FL = torch.Tensor([gamma_FL])
            self.use_scheduled_pos_weight = use_scheduled_pos_weight
            if use_scheduled_pos_weight:
                self.annihilation_speed = annihilation_speed
                self.pos_weight = torch.Tensor([pos_weight_0])
                self.pos_weight_inf = pos_weight
            if device is not None:
                self.gamma_FL = self.gamma_FL.to(device)
        self.gamma_bool = False
        if not use_focal_loss and np.abs(gamma - 1.) > 1e-3:
            print('')
            print("Will geometrically scale the positive weight.")
            self.gamma_bool = True
            self.gamma = torch.Tensor([gamma])
            print("gamma: "+str(gamma))
        if device is not None:
            self.pos_weight = self.pos_weight.to(device)
            if self. gamma_bool:
                self.gamma = self.gamma.to(device)

    def forward(self, scores, targets, cutoff=0.5, verbose=False, maxpool_gt=False, maxpool_prec=True):
        if self.use_indexed_peaks:
            peak_finding_mask = targets[:, 0, :, :].reshape(-1)
            indexing_mask = self.maxpool_idxg(targets)[:, 1, :, :].reshape(-1)
            rejected_mask = (peak_finding_mask + indexing_mask) % 2 # A XOR B
            intersection_mask = peak_finding_mask * indexing_mask # A AND B
            exclusion_mask = (1 - peak_finding_mask) * (1 - indexing_mask) # not A AND not B
            union_mask = peak_finding_mask + indexing_mask - intersection_mask # A OR B
            scores_filtered = scores[:, 0, :, :].reshape(-1)
            # scores_filtered[rejected_mask > 0.5] = -1e3 # predictions in A XOR B are artificially removed for the loss computation
            scores_filtered = scores_filtered[rejected_mask < 0.5] # predictions in A XOR B are  removed for the loss computation
            intersection_mask_filtered = intersection_mask[rejected_mask < 0.5] # targets in A XOR B are artificially removed for the loss computation

            if self.use_focal_loss:
                if self.use_scheduled_pos_weight:
                    self.pos_weight = update_schedule_pos_weight(self.pos_weight, self.pos_weight_inf, self.annihilation_speed)
                loss = focal_loss(scores_filtered, intersection_mask_filtered, exclusion_mask, self.pos_weight, self.gamma_FL)
            else:
                n_p = exclusion_mask.sum().double() / intersection_mask.sum().double()
                if self.gamma_bool:
                    n_p = n_p ** self.gamma
                pos_weight = self.pos_weight * n_p
                self.bceloss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                loss = self.bceloss(scores_filtered, intersection_mask_filtered) / self.pos_weight

            with torch.no_grad():
                n_pos_gt = intersection_mask.sum()
                n_neg_gt = exclusion_mask.sum()
                scores_c = scores[:, 0, :, :].reshape(-1)
                positives = (nn.Sigmoid()(scores_c) > cutoff)
                n_p = positives.sum()
                n_tp_recall = (positives * intersection_mask).sum()
                if maxpool_prec:
                    # maxpool for peak finding for precision only
                    peak_finding_mask_mp = self.maxpool(targets)[:, 0, :, :].reshape(-1)
                    intersection_mask_mp = peak_finding_mask_mp * indexing_mask
                    union_mask_mp = peak_finding_mask_mp + indexing_mask - intersection_mask_mp
                    n_tp_prec = (positives * union_mask_mp).sum()
                else:
                    n_tp_prec = (positives * union_mask).sum()
                recall = float(n_tp_recall) / max(1, int(n_pos_gt))
                precision = float(n_tp_prec) / max(1, int(n_p))
            metrics = {"loss": loss, "recall": recall, "precision": precision, "n_pos_gt": n_pos_gt.item(), "n_neg_gt": n_neg_gt.item()}
            return metrics

        else:
            if maxpool_gt:
                targets = self.maxpool(targets)
            scores_c = scores[:, 0, :, :].reshape(-1)
            targets_c = targets[:, 0, :, :].reshape(-1)
            gt_mask = targets_c > 0.5

            if self.use_focal_loss:
                loss = focal_loss(scores_c, targets_c, ~gt_mask, self.pos_weight, self.gamma_FL)
            else:
                n_p = (~gt_mask).sum().double() / gt_mask.sum().double() # negative over positive
                if self.gamma_bool:
                    n_p = n_p ** self.gamma
                pos_weight = self.pos_weight * n_p
                self.bceloss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                loss = self.bceloss(scores_c, targets_c) / self.pos_weight # the sigmoid is integrated to bceloss

            with torch.no_grad():
                n_gt = targets_c.sum()
                positives = (nn.Sigmoid()(scores_c) > cutoff)
                n_p = positives.sum()
                n_tp = (positives[gt_mask]).sum()
                if maxpool_prec:
                    n_tp_prec = (positives[self.maxpool(targets)[:, 0, :, :].reshape(-1) > 0.5]).sum()
                else:
                    n_tp_prec = n_tp
                recall = float(n_tp) / max(1, int(n_gt))
                precision = float(n_tp_prec) / max(1, int(n_p))
                if verbose:
                    print("nGT", int(n_gt), "recall", int(n_tp), "nP", int(n_p), "loss", float(loss.data))
            metrics = {"loss": loss, "recall": recall, "precision": precision}
            return metrics

class PeaknetMSELoss(nn.Module):
    
    def __init__(self, coor_scale=1, obj_scale=5, noobj_scale=1):
        super(PeaknetMSELoss, self).__init__()
        self.coor_scale = coor_scale
        self.obj_scale = obj_scale
        self.noobj_scale = noobj_scale
        
    def forward(self, scores, targets, cutoff=0.1, verbose=False):
        n = scores.size(0)
        m = scores.size(1)
        h = scores.size(2)
        w = scores.size(3)
        scores_c = nn.Sigmoid()(scores[:, 0, :, :].reshape(-1))
        targets_c = targets[:, 0, :, :].reshape(-1)
        if m > 1:
            scores_x = nn.Sigmoid()(scores[:, 2, :, :].reshape(-1))
            scores_y = nn.Sigmoid()(scores[:, 1, :, :].reshape(-1))
            targets_x = targets[:, 2, :, :].reshape(-1)
            targets_y = targets[:, 1, :, :].reshape(-1)
        loss_obj = self.obj_scale * nn.MSELoss()(scores_c[targets_c > 0.5], targets_c[targets_c > 0.5])
        loss_noobj = self.noobj_scale * nn.MSELoss()(scores_c[targets_c < 0.5], targets_c[targets_c < 0.5])
        if m > 1:
            loss_x = self.coor_scale * nn.MSELoss()(scores_x[targets_c > 0], targets_x[targets_c > 0])
            loss_y = self.coor_scale * nn.MSELoss()(scores_y[targets_c > 0], targets_y[targets_c > 0])
            loss = self.obj_scale * loss_obj + self.noobj_scale * loss_noobj + self.coor_scale * (loss_x + loss_y)
        else:
            loss = self.obj_scale * loss_obj + self.noobj_scale * loss_noobj
        with torch.no_grad():
            n_gt = targets_c.sum()
            mask = torch.zeros_like(scores_c)
            mask[scores_c > cutoff] = 1
            n_p = mask.sum()
            tp = torch.zeros_like(scores_c)
            tp[mask*targets_c > 0.5] = 1
            n_tp = tp.sum()
            recall = float(n_tp/n_gt)
            precision = float(n_tp/n_p)
            if verbose:
                if m > 1:
                    print("nGT", float(n_gt.data), "recall", float(n_tp.data), "loss", float(loss.data),
                          "x", float(loss_x.data), "y", float(loss_y.data),
                          "obj", float(loss_obj.data), "noobj", float(loss_noobj.data))
                else:
                    print("nGT", float(n_gt.data), "recall", float(n_tp.data), "loss", float(loss.data),
                          "obj", float(loss_obj.data), "noobj", float(loss_noobj.data))
        return loss, recall, precision
