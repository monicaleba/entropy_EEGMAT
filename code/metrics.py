"""Metric helpers: confusion-count metrics, subject-cluster bootstrap, information-theoretic metrics."""
import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score


def cm_metrics(c):
    """c [...,4] = TP, FN, FP, TN (positive = stress) -> dict of arrays"""
    TP, FN, FP, TN = [c[..., i].astype(float) for i in range(4)]
    n = TP + FN + FP + TN
    with np.errstate(all="ignore"):
        acc = (TP + TN) / n
        sens = np.where(TP + FN > 0, TP / (TP + FN), np.nan)
        spec = np.where(TN + FP > 0, TN / (TN + FP), np.nan)
        bacc = (sens + spec) / 2
        pe = ((TP + FN) * (TP + FP) + (FP + TN) * (FN + TN)) / n ** 2
        kappa = np.where(pe < 1, (acc - pe) / (1 - pe), 0.0)
        den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc = np.where(den > 0, (TP * TN - FP * FN) / den, 0.0)
        prec = np.where(TP + FP > 0, TP / (TP + FP), 0.0)
        f1 = np.where(2 * TP + FP + FN > 0, 2 * TP / (2 * TP + FP + FN), 0.0)
        prev = (TP + FN) / n
    return dict(acc=acc, bacc=bacc, kappa=kappa, mcc=mcc, sens=sens, spec=spec, prec=prec, f1=f1, prev=prev)


class AucBoot:
    """Weighted AUC for repeated bootstrap weight vectors; one instance per run."""
    def __init__(self, y, s, subj_col):
        self.y = y; self.subj = subj_col
        u, inv = np.unique(s, return_inverse=True)
        self.inv = inv; self.U = len(u)

    def __call__(self, Wsub):
        """Wsub [B, S] subject multiplicities -> AUC [B]"""
        w = Wsub[:, self.subj]                             # [B, n]
        pos = (self.y == 1); neg = ~pos
        B = w.shape[0]
        posw = np.zeros((B, self.U)); negw = np.zeros((B, self.U))
        for b in range(B):
            posw[b] = np.bincount(self.inv, weights=w[b] * pos, minlength=self.U)
            negw[b] = np.bincount(self.inv, weights=w[b] * neg, minlength=self.U)
        below = np.cumsum(negw, 1) - negw
        num = (posw * (below + 0.5 * negw)).sum(1)
        with np.errstate(all="ignore"):
            return num / (posw.sum(1) * negw.sum(1))


def run_counts(y, pred, subj_col, S):
    """per-subject confusion counts [S,4]"""
    out = np.zeros((S, 4))
    for code, (yy, pp) in enumerate([(1, 1), (1, 0), (0, 1), (0, 0)]):
        m = (y == yy) & (pred == pp)
        out[:, code] = np.bincount(subj_col[m], minlength=S)
    return out


def boot_metrics(runs, S, W, with_auc=True):
    """runs: list of dict(y, p1, subj_col). W [B,S] bootstrap multiplicities (row of ones = point estimate).
    Returns dict metric -> array [B] (mean over runs)."""
    B = W.shape[0]
    counts = np.stack([run_counts(r["y"], (r["p1"] > 0.5).astype(int), r["subj"], S) for r in runs])  # [R,S,4]
    tot = np.einsum("bs,rsc->brc", W, counts)               # [B,R,4]
    m = cm_metrics(tot)
    res = {k: np.nanmean(v, axis=1) for k, v in m.items()}
    if with_auc:
        aucs = np.stack([AucBoot(r["y"], r["p1"], r["subj"])(W) for r in runs], 1)
        res["auc"] = np.nanmean(aucs, axis=1)
    return res


# ---------------- information-theoretic metrics from a confusion matrix -------------------
def _h(p):
    p = np.asarray(p, float); p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def info_metrics(c):
    """c = [TP, FN, FP, TN]; positive = stress. Returns bits."""
    TP, FN, FP, TN = [float(x) for x in c]
    n = TP + FN + FP + TN
    J = np.array([[TN, FP], [FN, TP]]) / n                  # rows: true y (0,1) ; cols: predicted
    py = J.sum(1); pp = J.sum(0)
    Hy = _h(py); Hp = _h(pp); Hj = _h(J.ravel())
    I = Hy + Hp - Hj
    ent_norm = Hy / np.log2(2)                              # H(Y)/log2|Y|  (|Y| = 2)
    # label vectors for sklearn NMI / AMI
    yt = np.repeat([0, 0, 1, 1], [int(TN), int(FP), int(FN), int(TP)])
    yp = np.repeat([0, 1, 0, 1], [int(TN), int(FP), int(FN), int(TP)])
    nmi = normalized_mutual_info_score(yt, yp, average_method="geometric")
    ami = adjusted_mutual_info_score(yt, yp) if len(set(yp)) > 1 else 0.0
    # confusion entropy (Wei et al. 2010), K = 2 -> log base 2
    C = np.array([[TN, FP], [FN, TP]])
    cen = 0.0
    for j in range(2):
        k = 1 - j
        den = C[j].sum() + C[:, j].sum()                   # sum_l (C_jl + C_lj), diagonal included
        ps = [C[j, k] / den, C[k, j] / den] if den > 0 else []
        cen_j = -sum(q * np.log2(q) for q in ps if q > 0)
        cen += den / (2 * C.sum()) * cen_j
    return dict(Hy=Hy, Hy_norm=ent_norm, I=I, I_over_H=I / Hy if Hy > 0 else 0.0, H_cond=Hy - I,
                nmi=nmi, ami=ami, cen=cen)


# ---------------- spectral entropy -------------------
def spectral(W):
    sv = np.linalg.svd(W, compute_uv=False)
    p = sv / sv.sum()
    H = -(p[p > 0] * np.log2(p[p > 0])).sum()
    er = np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())
    return H, er, er / min(W.shape)
