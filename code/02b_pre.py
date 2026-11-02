"""Step 2b: 'PRE' condition = the ORIGINAL submission's order of operations, kept only as a control:
balance the whole pool (Dummy duplicates / augmented copies) FIRST, then split randomly 80/20 at
epoch level. Test items may therefore be duplicates/augmented copies, and their source epoch (or
an identical copy) may sit in the training partition.  10 repeats per cell, validation = 15% of train.
Stores per-run test predictions plus the fraction of test items whose source epoch is also in train.
"""
import os, sys, pickle, zlib, time
import numpy as np
import torch
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
from sklearn.model_selection import train_test_split
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results"); os.makedirs(os.path.join(RES, "runs"), exist_ok=True)
D = np.load(os.path.join(RES, "features.npz"))
fs = "msk19"


def feats(arr):
    a = arr[..., :19, :][..., [0, 1, 2]]
    return a.reshape(a.shape[:-2] + (-1,)).astype(np.float32)


def cell(args):
    sub, mode = args
    torch.set_num_threads(3)
    Xr = feats(D["X"]); Xa = feats(D["Xa"]); y = D["y"]
    I = np.arange(len(y))
    if sub == "good": I = I[D["good"] == 1]
    if sub == "bad": I = I[D["good"] == 0]
    data, meta = [], []
    for r in range(10):
        rng = np.random.default_rng(500 + r)
        yy = y[I]; n0, n1 = (yy == 0).sum(), (yy == 1).sum()
        mi = I[yy == 1] if n1 < n0 else I[yy == 0]; mlab = 1 if n1 < n0 else 0
        need = abs(n0 - n1)
        if mode == "dummy":
            src = rng.choice(mi, need, replace=True); Xn = Xr[src]
        else:
            pairs = np.array([(i, a) for i in mi for a in range(Xa.shape[1])])
            sel = pairs[rng.choice(len(pairs), need, replace=need > len(pairs))]
            src = sel[:, 0]; Xn = np.stack([Xa[i, a] for i, a in sel])
        Xp = np.concatenate([Xr[I], Xn]); yp = np.concatenate([y[I], np.full(need, mlab)])
        sp = np.concatenate([I, src]); subp = D["subj"][sp]
        tr, te = train_test_split(np.arange(len(yp)), test_size=0.2, stratify=yp, random_state=int(rng.integers(1 << 30)))
        tr2, va = train_test_split(tr, test_size=0.15, stratify=yp[tr], random_state=int(rng.integers(1 << 30)))
        mu = Xp[tr2].mean(0); sd = Xp[tr2].std(0) + 1e-8
        z = lambda a: ((a - mu) / sd).astype(np.float32)
        data.append(dict(Xtr=z(Xp[tr2]), ytr=yp[tr2], Xva=z(Xp[va]), yva=yp[va], Xte=z(Xp[te]), yte=yp[te]))
        trsrc = set(sp[tr2]) | set(sp[va])
        meta.append(dict(y=yp[te], subj=subp[te], twin=np.array([s in trsrc for s in sp[te]]), nte=len(te)))
    t0 = time.time()
    out = engine.train_models(data, epochs=1000, seed=zlib.crc32(f"pre{sub}{mode}".encode()) % 100000)
    out["meta"] = meta
    pickle.dump(out, open(os.path.join(RES, "runs", f"pre_{sub}_{mode}.pkl"), "wb"))
    print("done", sub, mode, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    jobs = [(s, m) for s in ("full", "good", "bad") for m in ("dummy", "aug")]
    with Pool(6) as p:
        p.map(cell, jobs)
