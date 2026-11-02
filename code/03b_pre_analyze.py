"""Analyse the PRE control (balance-then-split, original submission order): mean +- SD over 10 repeats."""
import os, sys, pickle
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
rows = []
for sub in ("full", "good", "bad"):
    for mode in ("dummy", "aug"):
        o = pickle.load(open(os.path.join(RES, "runs", f"pre_{sub}_{mode}.pkl"), "rb"))
        for which in ("sel", "fin"):
            ms = []
            for r, m in enumerate(o["meta"]):
                p1 = o["probs_" + which][r][:, 1]; y = m["y"]
                pred = (p1 > .5).astype(int)
                c = np.array([[((y == 1) & (pred == 1)).sum(), ((y == 1) & (pred == 0)).sum(),
                               ((y == 0) & (pred == 1)).sum(), ((y == 0) & (pred == 0)).sum()]])
                mm = M.cm_metrics(c); d = {k: float(v[0]) for k, v in mm.items()}
                d["auc"] = roc_auc_score(y, p1); d["twin"] = m["twin"].mean(); d["n"] = m["nte"]
                ms.append(d)
            df = pd.DataFrame(ms)
            row = dict(subset=sub, mode=mode, which=which)
            for k in ("acc", "bacc", "kappa", "mcc", "auc", "twin", "n", "prev"):
                row[k] = df[k].mean(); row[k + "_sd"] = df[k].std()
            rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(RES, "pre_summary.csv"), index=False)
print(pd.DataFrame(rows).round(3).to_string())

# accuracy on test items that do / do not have a twin (source epoch or copy) in the training data
rows = []
for sub in ("full", "good", "bad"):
    for mode in ("dummy", "aug"):
        o = pickle.load(open(os.path.join(RES, "runs", f"pre_{sub}_{mode}.pkl"), "rb"))
        a_t, a_n = [], []
        for r, m in enumerate(o["meta"]):
            ok = (o["probs_sel"][r][:, 1] > .5).astype(int) == m["y"]
            a_t.append(ok[m["twin"]].mean()); a_n.append(ok[~m["twin"]].mean())
        rows.append(dict(subset=sub, mode=mode, acc_twin=np.mean(a_t), acc_no_twin=np.mean(a_n)))
pd.DataFrame(rows).to_csv(os.path.join(RES, "pre_twin_split.csv"), index=False)
