"""Step 2: run the full experimental grid.

Grid for one (feature set) :  3 subsets (Full, Good, Bad)  x  3 training-set modes
(raw, Dummy = random oversampling by duplication, Aug = noise/scale/shift augmentation)
x 4 evaluation protocols:
  RAND  : random stratified 80/20 epoch split, 10 repeats (literature-style, subject-dependent)
  BLOCK : within-subject chronologically blocked 5-fold CV (1-epoch purge gap), 2 seeds
  LOSO  : leave-one-subject-out, 3 seeds (subject-independent, main protocol)
  GS80  : repeated subject-grouped 80/20 splits, 20 repeats (the design of the original submission)

Balancing (Dummy/Aug) is applied ONLY to the training part of every split; validation and
test partitions always keep their natural class distribution and contain only real epochs.
"""
import os, sys, pickle, time, itertools, zlib
import numpy as np
import torch
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
D = np.load(os.path.join(RES, "features.npz"))
STAT_IDX = {"mean": 0, "std": 1, "skew": 2, "kurt": 3, "rms": 4}
FEATSETS = {  # name: (channel slice, stats)
    "msk19": (slice(0, 19), ["mean", "std", "skew"]),           # primary: 19 scalp EEG channels
    "msr19": (slice(0, 19), ["mean", "std", "rms"]),            # original statistics, 19 EEG channels
    "msr21": (slice(0, 21), ["mean", "std", "rms"]),            # original submission (EEG + A2-A1 + ECG)
    "mskk19": (slice(0, 19), ["mean", "std", "skew", "kurt"]),  # 4 moments
}


def feats(arr, fs):
    sl, st = FEATSETS[fs]
    a = arr[..., sl, :][..., [STAT_IDX[s] for s in st]]
    return a.reshape(a.shape[:-2] + (-1,)).astype(np.float32)


def rand_split(idx_y, rng, test_frac=0.2):
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(idx_y)), test_size=test_frac, stratify=idx_y,
                              random_state=int(rng.integers(1 << 30)))
    return tr, te


def make_folds(subset_idx, proto):
    """Returns list of dict(tr,va,te,meta) with indices into the global epoch arrays."""
    subj, y, ep = D["subj"], D["y"], D["eidx"]
    I = subset_idx
    subs = np.unique(subj[I])
    folds = []
    if proto == "RAND":
        for r in range(10):
            rng = np.random.default_rng(1000 + r)
            tr, te = rand_split(y[I], rng, 0.2)
            tr2, va = rand_split(y[I][tr], rng, 0.15)
            folds.append(dict(tr=I[tr][tr2], va=I[tr][va], te=I[te], meta=("RAND", r, -1)))
    elif proto == "BLOCK":
        for f in range(5):
            tr, va, te = [], [], []
            for s in subs:
                for lab in (0, 1):
                    ii = I[(subj[I] == s) & (y[I] == lab)]
                    ii = ii[np.argsort(ep[ii])]
                    blocks = np.array_split(np.arange(len(ii)), 5)
                    tpos = blocks[f]
                    lo, hi = tpos[0] - 1, tpos[-1] + 1             # purge gap of one epoch
                    pool = [k for k in range(len(ii)) if k < lo or k > hi]
                    nv = max(1, int(round(0.15 * len(pool))))
                    vpos = pool[-nv:]
                    trpos = [k for k in pool[:-nv] if k < vpos[0] - 1]  # purge between train and val
                    te += list(ii[tpos]); va += list(ii[vpos]); tr += list(ii[trpos])
            folds.append(dict(tr=np.array(tr), va=np.array(va), te=np.array(te), meta=("BLOCK", f, -1)))
    elif proto == "LOSO":
        for seed in range(3):
            rng = np.random.default_rng(2000 + seed)
            for s in subs:
                rest = [t for t in subs if t != s]
                nv = max(2, int(round(0.15 * len(rest))))
                vs = rng.choice(rest, nv, replace=False)
                tr_s = [t for t in rest if t not in vs]
                folds.append(dict(tr=I[np.isin(subj[I], tr_s)], va=I[np.isin(subj[I], vs)],
                                  te=I[subj[I] == s], meta=("LOSO", seed, int(s))))
    elif proto == "GS80":
        for r in range(20):
            rng = np.random.default_rng(3000 + r)
            nt = max(2, int(round(0.2 * len(subs))))
            ts = rng.choice(subs, nt, replace=False)
            rest = np.array([t for t in subs if t not in ts])
            nv = max(2, int(round(0.15 * len(rest))))
            vs = rng.choice(rest, nv, replace=False)
            tr_s = [t for t in rest if t not in vs]
            folds.append(dict(tr=I[np.isin(subj[I], tr_s)], va=I[np.isin(subj[I], vs)],
                              te=I[np.isin(subj[I], ts)], meta=("GS80", r, -1)))
    return folds


def balance(tr, mode, rng, Xr, Xa):
    """Return (X, y) of the training set after (optional) balancing. Only TRAIN epochs are used."""
    y = D["y"]
    X = Xr[tr]; yy = y[tr]
    if mode == "raw":
        return X, yy
    n0, n1 = (yy == 0).sum(), (yy == 1).sum()
    minority = 1 if n1 < n0 else 0
    need = abs(n0 - n1)
    mi = tr[yy == minority]
    if mode == "dummy":
        pick = rng.choice(mi, need, replace=True)
        return np.concatenate([X, Xr[pick]]), np.concatenate([yy, np.full(need, minority)])
    if mode == "aug":
        A = Xa.shape[1]
        pairs = np.array([(i, a) for i in mi for a in range(A)])
        sel = pairs[rng.choice(len(pairs), need, replace=need > len(pairs))]
        Xaug = np.stack([Xa[i, a] for i, a in sel])
        return np.concatenate([X, Xaug]), np.concatenate([yy, np.full(need, minority)])
    raise ValueError(mode)


def run_cell(args):
    fs, subset, mode, epochs, protos = args
    torch.set_num_threads(3)
    name = f"{fs}_{subset}_{mode}"
    outp = os.path.join(RES, "runs", name + ".pkl")
    if os.path.exists(outp):
        return name
    good = D["good"]
    I = np.arange(len(D["y"]))
    if subset == "good":
        I = I[good == 1]
    elif subset == "bad":
        I = I[good == 0]
    Xr = feats(D["X"], fs); Xa = feats(D["Xa"], fs)
    folds = []
    for pr in protos:
        folds += make_folds(I, pr)
    data = []
    for k, f in enumerate(folds):
        rng = np.random.default_rng(7 + k)
        Xt, yt = balance(f["tr"], mode, rng, Xr, Xa)
        mu = Xr[f["tr"]].mean(0); sd = Xr[f["tr"]].std(0) + 1e-8      # scaler fit on real training epochs only
        z = lambda a: ((a - mu) / sd).astype(np.float32)
        data.append(dict(Xtr=z(Xt), ytr=yt, Xva=z(Xr[f["va"]]), yva=D["y"][f["va"]],
                         Xte=z(Xr[f["te"]]), yte=D["y"][f["te"]], mu=mu, sd=sd))
    t0 = time.time()
    out = engine.train_models(data, epochs=epochs, seed=zlib.crc32(name.encode()) % 100000)
    out["folds"] = [dict(te=f["te"], va=f["va"], tr=f["tr"], meta=f["meta"]) for f in folds]
    out["scaler"] = [(d["mu"], d["sd"]) for d in data]
    out["ntrain"] = [len(d["ytr"]) for d in data]
    out["mode"] = mode; out["subset"] = subset; out["fs"] = fs
    os.makedirs(os.path.join(RES, "runs"), exist_ok=True)
    with open(outp, "wb") as fh:
        pickle.dump(out, fh)
    print(f"done {name}: {len(folds)} models, {time.time()-t0:.0f}s", flush=True)
    return name


if __name__ == "__main__":
    fs_list = sys.argv[1].split(",")
    epochs = int(sys.argv[2])
    protos = sys.argv[3].split(",")
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    jobs = [(fs, s, m, epochs, protos) for fs in fs_list for s in ("full", "good", "bad")
            for m in ("raw", "dummy", "aug")]
    with Pool(nproc) as pool:
        for n in pool.imap_unordered(run_cell, jobs):
            pass
