"""Step 4: SHAP explanation entropy for validated (best-validation-epoch) models.
For every (subset, mode) cell and for the protocols LOSO and RAND, 6 models are explained
(60 test rows, 50 background rows from the model's own scaled training set); the class-'stress'
probability is explained with shap.Explainer (permutation algorithm).
"""
import os, sys, pickle
import numpy as np, pandas as pd, shap
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
fs = sys.argv[1] if len(sys.argv) > 1 else "msk19"
D = np.load(os.path.join(RES, "features.npz"))
STAT_IDX = {"mean": 0, "std": 1, "skew": 2, "kurt": 3, "rms": 4}
FEATSETS = {"msk19": (slice(0, 19), ["mean", "std", "skew"]), "msr19": (slice(0, 19), ["mean", "std", "rms"]),
            "msr21": (slice(0, 21), ["mean", "std", "rms"]), "mskk19": (slice(0, 19), ["mean", "std", "skew", "kurt"])}


def feats(arr):
    sl, stt = FEATSETS[fs]
    a = arr[..., sl, :][..., [STAT_IDX[s] for s in stt]]
    return a.reshape(a.shape[:-2] + (-1,)).astype(np.float32)


def job(args):
    sub, mode, proto = args
    out = pickle.load(open(os.path.join(RES, "runs", f"{fs}_{sub}_{mode}.pkl"), "rb"))
    Xr = feats(D["X"])
    ks = [k for k, f in enumerate(out["folds"]) if f["meta"][0] == proto and (proto != "LOSO" or f["meta"][1] == 0)][:6]
    ent_f, ent_c, evn_f, evn_c, top1, ps, imp_all = [], [], [], [], [], [], []
    rng = np.random.default_rng(42)
    nstat = len(FEATSETS[fs][1]); nch = Xr.shape[1] // nstat
    for k in ks:
        mu, sd = out["scaler"][k]
        tr = out["folds"][k]["tr"]; te = out["folds"][k]["te"]
        bg = ((Xr[rng.choice(tr, 50, replace=False)] - mu) / sd).astype(np.float64)
        ex = ((Xr[rng.choice(te, min(60, len(te)), replace=False)] - mu) / sd).astype(np.float64)
        f = lambda X, k=k: engine.numpy_forward(out["par_sel"], out["run_sel"], k, X.astype(np.float32))
        expl = shap.Explainer(f, bg, algorithm="permutation")
        sv = expl(ex, max_evals=2 * Xr.shape[1] * 8 + 1, silent=True).values        # [60, F]
        a = np.abs(sv)
        m = a.mean(0); p = m / m.sum()
        # features are laid out channel-major: reshape(ch, stat)
        c = m.reshape(nch, nstat).sum(1); pc = c / c.sum()
        H = lambda q: -(q[q > 0] * np.log2(q[q > 0])).sum()
        ent_f.append(H(p)); ent_c.append(H(pc)); evn_f.append(H(p) / np.log2(len(p))); evn_c.append(H(pc) / np.log2(len(pc)))
        top1.append(p.max())
        rows = a / a.sum(1, keepdims=True)
        ps.append(np.mean([H(r) for r in rows]))
        imp_all.append(p)
    p_mean = np.mean(imp_all, 0)
    return dict(subset=sub, mode=mode, proto=proto, feat_H=np.mean(ent_f), feat_even=np.mean(evn_f),
                eff_feat=2 ** np.mean(ent_f), top1=np.mean(top1), chan_H=np.mean(ent_c), chan_even=np.mean(evn_c),
                eff_chan=2 ** np.mean(ent_c), per_sample_H=np.mean(ps), per_sample_sd=np.std(ps), imp=p_mean.tolist(), n=len(ks))


if __name__ == "__main__":
    jobs = [(s, m, p) for s in ("full", "good", "bad") for m in ("raw", "dummy", "aug") for p in ("LOSO", "RAND")]
    with Pool(8) as pool:
        rows = pool.map(job, jobs)
    pickle.dump(rows, open(os.path.join(RES, f"shap_{fs}.pkl"), "wb"))
    pd.DataFrame([{k: v for k, v in r.items() if k != "imp"} for r in rows]).to_csv(os.path.join(RES, f"shap_{fs}.csv"), index=False)
    print("shap done")
