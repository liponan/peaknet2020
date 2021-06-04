from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
import h5py
import json
import psana


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

    def __init__(self, cxi_path, exp, run, normalize=False, downsample=1, debug=True,
                 max_cutoff=1024, mode="peaknet2020", shuffle=False, n=-1, min_det_peaks=-1):
        self.downsample = downsample
        self.cxi = CXILabel(cxi_path)
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
                label[i, 0, u, v] = 1
                label[i, 1, u, v] = np.fmod(my_r[j] / float(self.downsample), 1.0) #
                label[i, 2, u, v] = np.fmod(my_c[j] / float(self.downsample), 1.0)
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
        event_idx, s, r, c = self.cxi[self.rand_idxs[idx]]
        n_trials = 1
        while len(s) < self.min_det_peaks:
            idx = (idx + 1) % self.n
            n_trials += 1
            event_idx, s, r, c = self.cxi[self.rand_idxs[idx]]
        img = self.psana.load_img(event_idx)
        img[img < 0] = 0
        if self.normalize:
            img = img / max(0.01, np.std(img)) # why max 0.01?
            img = img - np.mean(img)
            # img = img / max(np.max(img), self.max_cutoff)
        h_ds = int(np.ceil(img.shape[1] / float(self.downsample)))
        w_ds = int(np.ceil(img.shape[2] / float(self.downsample)))
        h_pad = int(h_ds * self.downsample)
        w_pad = int(w_ds * self.downsample)
        if self.mode == "peaknet2020":
            img_tensor = torch.zeros(img.shape[0], h_pad, w_pad)
            img_tensor[:, 0:img.shape[1], 0:img.shape[2]] = torch.from_numpy(img)
            label_tensor = self.make_label(s, r, c, n_panels=img.shape[0], h=h_ds, w=w_ds)
            n_trials_tensor = torch.zeros(1)
            n_trials_tensor[0] = n_trials
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
    
    def __init__(self, cxi_path, fmod=True):
        self.f = h5py.File(cxi_path, "r")
        self.nPeaks = self.f["entry_1/result_1/nPeaks"]
        self.n_hits = len(self.nPeaks)
        self.eventIdx = self.f["LCLS/eventNumber"][:self.n_hits]
        self.peak_x_label = self.f['entry_1/result_1/peakXPosRaw'][:self.n_hits, :]
        self.peak_y_label = self.f['entry_1/result_1/peakYPosRaw'][:self.n_hits, :]
        #self.peak_x_center = self.f['entry_1/result_1/peakXPosRaw'][:self.n_hits, :]
        #self.peak_y_center = self.f['entry_1/result_1/peakYPosRaw'][:self.n_hits, :]
        self.peak_x_center = self.f['entry_1/result_1/peak2'][:self.n_hits, :]
        self.peak_y_center = self.f['entry_1/result_1/peak1'][:self.n_hits, :]
        #self.peak_w = self.f['entry_1/result_1/peak4'][:self.n_hits, :]
        #self.peak_h = self.f['entry_1/result_1/peak3'][:self.n_hits, :]
        self.detector = str(self.f["entry_1/instrument_1/detector_1/description"][()])
        
    def __len__(self):
        return self.n_hits
        
    def __getitem__(self, idx):
        my_npeaks = self.nPeaks[idx]
        my_event_idx = self.eventIdx[idx]
        # psana style
        my_s = np.floor_divide(self.peak_y_label[idx, 0:my_npeaks], 185) \
            + 8 * np.floor_divide(self.peak_x_label[idx, 0:my_npeaks], 388)
        my_r = np.fmod(self.peak_y_center[idx, 0:my_npeaks], 185.0)
        my_c = np.fmod(self.peak_x_center[idx, 0:my_npeaks], 388.0)
        return my_event_idx, my_s, my_r, my_c
        # psocake style
#         my_r = self.peak_y_center[idx,0:my_npeaks]
#         my_c = self.peak_x_center[idx,0:my_npeaks]
#         return (my_event_idx, my_r, my_c)

    def close(self):
        self.f.close()
