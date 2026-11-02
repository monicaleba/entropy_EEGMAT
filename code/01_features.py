"""Step 1: read EEGMAT EDFs, band-pass 0.5-45 Hz, cut 4-s epochs, compute per-channel
time-domain statistics (mean, std, skewness, excess kurtosis, RMS) for every epoch and
a bank of augmented copies of every epoch (used later ONLY inside training partitions).

Output: results/features.npz
"""
import os, glob, sys
import numpy as np, pandas as pd, mne
from scipy.stats import skew, kurtosis

mne.set_log_level("ERROR")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "eeg-mental")
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

FS = 500
EP = 4 * FS                  # 4-s epoch
N_AUG = 6                    # augmented copies stored per epoch
STATS = ["mean", "std", "skew", "kurt", "rms"]


def stats(x):
    """x: [..., C, N] -> [..., C, 5]"""
    m = x.mean(-1)
    s = x.std(-1)                      # population std (ddof=0)
    sk = skew(x, axis=-1)
    ku = kurtosis(x, axis=-1)          # excess kurtosis
    rms = np.sqrt((x ** 2).mean(-1))
    return np.stack([m, s, sk, ku, rms], -1)


info = pd.read_csv(os.path.join(DATA, "subject-info.csv"))
rng = np.random.default_rng(42)

X, Xa, subj, lab, eidx, ngrp = [], [], [], [], [], []
ch_names = None
for si in range(36):
    for rec in (1, 2):
        f = os.path.join(DATA, f"Subject{si:02d}_{rec}.edf")
        raw = mne.io.read_raw_edf(f, preload=True)
        assert raw.info["sfreq"] == FS
        raw.filter(0.5, 45.0, fir_design="firwin", verbose=False)   # MNE default FIR band-pass
        d = raw.get_data() * 1e6                                      # microvolts
        ch_names = [c.replace("EEG ", "") for c in raw.ch_names]
        sig_sd = d.std(axis=1)                                        # per-channel SD of recording
        n_ep = d.shape[1] // EP
        for e in range(n_ep):
            seg = d[:, e * EP:(e + 1) * EP]
            X.append(stats(seg))
            copies = []
            for a in range(N_AUG):
                s = rng.uniform(0.9, 1.1, size=(seg.shape[0], 1))
                shift = rng.integers(-int(0.05 * EP), int(0.05 * EP) + 1)
                noise = rng.normal(0.0, 1.0, size=seg.shape) * (0.05 * sig_sd[:, None])
                aug = s * np.roll(seg, shift, axis=1) + noise
                copies.append(stats(aug))
            Xa.append(np.stack(copies))
            subj.append(si); lab.append(rec - 1); eidx.append(e)
            ngrp.append(int(info.loc[si, "Count quality"]))
    print("subject", si, "done", flush=True)

X = np.array(X, dtype=np.float32); Xa = np.array(Xa, dtype=np.float32)
np.savez_compressed(os.path.join(OUT, "features.npz"), X=X, Xa=Xa, subj=np.array(subj),
                    y=np.array(lab), eidx=np.array(eidx), good=np.array(ngrp),
                    ch=np.array(ch_names), stats=np.array(STATS))
print(X.shape, Xa.shape, "epochs per class:", np.bincount(lab), "channels:", ch_names)
