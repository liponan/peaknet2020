from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
import h5py
import json
import psana
import time

class PSANADataset(Dataset):

    def __init__(self, df_path, subset="train", n=-1, shuffle=False):
        self.df = pd.read_csv(df_path).query("subset == '{}'".format(subset))
        if n > 0:
            n = min(n, len(self.df))
            self.df = self.df.sample(n=n)
        if shuffle:
            self.df = self.df.sample(frac=1.0)
        self.n = len(self.df)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        file_path, exp, run = self.df.iloc[idx][["path", "exp", "run"]]
        return file_path, exp, run


class PSANAImage(Dataset):

    def __init__(self, cxi_path, exp, run, normalize=True, downsample=1, debug=True,
                 max_cutoff=1024, mode="peaknet2020", shuffle=False, n=-1, min_det_peaks=-1, use_indexed_peaks=False,
                 n_classes=3):
        self.use_indexed_peaks = use_indexed_peaks
        self.n_classes = n_classes
        self.downsample = downsample
        self.cxi = CXILabel(cxi_path, use_indexed_peaks)
        self.detector = self.cxi.detector  # "CxiDs2.0:Cspad.0"#
        if n == -1:
            self.n = len(self.cxi)
        else:
            self.n = min(n, len(self.cxi))
        self.normalize = normalize
        self.max_cutoff = max_cutoff
        self.debug = debug
        self.psana = PSANAReader(exp, run, self.detector)
        self.psana.build()
        self.mode = mode
        self.min_det_peaks = min_det_peaks
        if shuffle:
            self.rand_idxs = np.random.permutation(len(self.cxi))
            self.rand_idxs = self.rand_idxs[:self.n]
        else:
            self.rand_idxs = np.arange(self.n)

    def __len__(self):
        return self.n

    def make_label(self, s, r, c, n_panels=32, h=24, w=49):
        label = torch.zeros(n_panels, 3, h, w)
        for i in range(n_panels):
            my_r = r[s == i]
            my_c = c[s == i]
            for j in range(len(my_r)):
                u = int(np.floor(my_r[j] / float(self.downsample)))
                v = int(np.floor(my_c[j] / float(self.downsample)))
                if u < h and v < w:
                    label[i, 0, u, v] = 1
                    label[i, 1, u, v] = np.fmod(my_r[j] / float(self.downsample), 1.0) #
                    label[i, 2, u, v] = np.fmod(my_c[j] / float(self.downsample), 1.0)
        return label

    def make_label_with_idxg(self, s, r, c, s_idxg, r_idxg, c_idxg, n_panels=32, h=24, w=49):
        label = torch.zeros(n_panels, 6, h, w)
        label_peaks_only = self.make_label(s, r, c, n_panels=n_panels, h=h, w=w)
        label_idxg_only = self.make_label(s_idxg, r_idxg, c_idxg, n_panels=n_panels, h=h, w=w)
        label[:, :3, :, :] = label_peaks_only
        label[:, 3:, :, :] = label_idxg_only
        return label

    def make_yolo_labels(self, s, r, c, h_obj=7, w_obj=7):
        n = r.shape[0]
        cls = np.zeros((n,))
        ww = w_obj * np.ones((n,))
        hh = h_obj * np.ones((n,))
        return [cls, s, r, c, hh, ww]

    def close(self):
        self.cxi.close()
        self.psana.ds = None

    def __getitem__(self, idx):
        if self.use_indexed_peaks:
            event_idx, s, r, c, s_idxg, r_idxg, c_idxg = self.cxi[self.rand_idxs[idx]]
        else:
            event_idx, s, r, c = self.cxi[self.rand_idxs[idx]]
        n_trials = 1
        while len(s) < self.min_det_peaks:
            idx = (idx + 1) % self.n
            n_trials += 1
            if self.use_indexed_peaks:
                event_idx, s, r, c, s_idxg, r_idxg, c_idxg = self.cxi[self.rand_idxs[idx]]
            else:
                event_idx, s, r, c = self.cxi[self.rand_idxs[idx]]
        img = self.psana.load_img(event_idx)
        img[img < 0] = 0
        if self.normalize:
            # img = img / max(0.01, np.std(img)) # why max 0.01?
            # img = img - np.mean(img)
            for i in range(img.shape[0]):
                img[i] = img[i] / np.max(img[i])
        # h_ds = int(np.ceil(img.shape[1] / float(self.downsample)))
        # w_ds = int(np.ceil(img.shape[2] / float(self.downsample)))
        # h_pad = int(h_ds * self.downsample)
        # w_pad = int(w_ds * self.downsample)
        h = img.shape[1]
        w = img.shape[2]
        h_ds = int(img.shape[1] / float(self.downsample))
        w_ds = int(img.shape[2] / float(self.downsample))
        if self.mode == "peaknet2020":
            # img_tensor = torch.zeros(img.shape[0], h_pad, w_pad)
            img_tensor = torch.zeros(img.shape[0], h, w)
            img_tensor[:, 0:img.shape[1], 0:img.shape[2]] = torch.from_numpy(img)
            if self.use_indexed_peaks:
                label_tensor = self.make_label_with_idxg(s, r, c, s_idxg, r_idxg, c_idxg, n_panels=img.shape[0], h=h_ds, w=w_ds)
            else:
                label_tensor = self.make_label(s, r, c, n_panels=img.shape[0], h=h_ds, w=w_ds)
            n_trials_tensor = torch.zeros(1)
            n_trials_tensor[0] = n_trials
            if self.n_classes == 1:
                if self.use_indexed_peaks:
                    label_tensor = label_tensor[:, [0, 3], :, :]
                else:
                    label_tensor = label_tensor[:, 0:1, :, :]
            return img_tensor, label_tensor, n_trials_tensor
        else:  # YOLO mode
            labels = self.make_yolo_labels(s, r, c)
            return img, labels


class PSANAReader(object):

    def __init__(self, exp, run, det_name="DsdCsPad"):
        self.exp = exp
        self.run = run
        self.det_name = det_name
        self.ds = None
        self.det = None
        self.this_run = None
        self.times = None

    def build(self):
        self.ds = psana.DataSource("exp={}:run={}:idx".format(self.exp, self.run))
        self.det = psana.Detector(self.det_name)
        self.this_run = self.ds.runs().next()
        self.times = self.this_run.times()

    def load_img(self, event_idx):
        evt = self.this_run.event(self.times[event_idx])
        calib = self.det.calib(evt) * self.det.mask(evt, calib=True, status=True, edges=True,
                                                    central=True, unbond=True, unbondnbrs=True)
        return calib


class CXILabel(Dataset):

    def __init__(self, cxi_path, use_indexed_peaks, fmod=True):
        self.f = h5py.File(cxi_path, "r")
        # Test constistency
        # self.f_test = h5py.File('/cds/data/psdm/cxi/cxic0415/scratch/axlevy/psocake/r0100/cxic0415_0100_psocake2.cxi', "r")
        # self.nPeaks_test = self.f_test["entry_1/result_1/nPeaks"]
        # self.n_hits_test = len(self.nPeaks_test)
        # self.eventIdx_test = self.f_test["LCLS/eventNumber"][:self.n_hits_test]
        # print('eventIdx_test')
        # print(self.eventIdx_test)

        self.nPeaks = self.f["entry_1/result_1/nPeaks"]
        self.n_hits = len(self.nPeaks)
        self.eventIdx = self.f["LCLS/eventNumber"][:self.n_hits]
        self.peak_x_label = self.f['entry_1/result_1/peakXPosRaw'][:self.n_hits, :]
        self.peak_y_label = self.f['entry_1/result_1/peakYPosRaw'][:self.n_hits, :]
        self.peak_x_center = self.f['entry_1/result_1/peak2'][:self.n_hits, :]
        self.peak_y_center = self.f['entry_1/result_1/peak1'][:self.n_hits, :]
        self.detector = str(self.f["entry_1/instrument_1/detector_1/description"][()])
        # print('eventIdx')
        # print(self.eventIdx)

        self.use_indexed_peaks = use_indexed_peaks
        if use_indexed_peaks:
            self.nIndexedPeaks = self.f["indexing/nIndexedPeaks"]
            self.indexing_x_center = self.f["indexing/XPos"][:self.n_hits, :]
            self.indexing_y_center = self.f["indexing/YPos"][:self.n_hits, :]
            self.indexing_panel = self.f["indexing/panel"][:self.n_hits, :]

    def __len__(self):
        return self.n_hits

    def __getitem__(self, idx):
        my_npeaks = self.nPeaks[idx]
        my_event_idx = self.eventIdx[idx]
        # Test consistency
        # print(idx)
        # print(my_event_idx)
        # print(my_npeaks)
        # print("")
        # print(np.argwhere(self.eventIdx_test == my_event_idx))
        # idx_test = np.argwhere(self.eventIdx_test == my_event_idx)[0,0]
        # my_npeaks_test = self.nPeaks_test[idx_test]
        # my_event_idx_test = self.eventIdx_test[idx_test]
        # print(idx_test)
        # print(my_event_idx_test)
        # print(my_npeaks_test)
        # time.sleep(5)

        # psana style
        my_s = np.floor_divide(self.peak_y_label[idx, 0:my_npeaks], 185) \
            + 8 * np.floor_divide(self.peak_x_label[idx, 0:my_npeaks], 388)
        my_r = np.fmod(self.peak_y_center[idx, 0:my_npeaks], 185.0)
        my_c = np.fmod(self.peak_x_center[idx, 0:my_npeaks], 388.0)
        if self.use_indexed_peaks:
            my_nIndexedPeaks = self.nIndexedPeaks[idx]
            my_s_indexing = self.indexing_panel[idx, 0:my_nIndexedPeaks].astype(np.int32)
            my_r_indexing = np.fmod(self.indexing_y_center[idx, 0:my_nIndexedPeaks], 185.0)
            my_c_indexing = np.fmod(self.indexing_x_center[idx, 0:my_nIndexedPeaks], 388.0)
            return my_event_idx, my_s, my_r, my_c, my_s_indexing, my_r_indexing, my_c_indexing
        else:
            return my_event_idx, my_s, my_r, my_c

    def close(self):
        self.f.close()
