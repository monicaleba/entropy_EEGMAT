"""Step 3: aggregate runs -> tables (point estimates + subject-cluster bootstrap CIs), interaction tests,
per-subject analysis, information-theoretic metrics, training dynamics, spectral entropy.
usage: python 03_analyze.py <featset>
"""
import os, sys, pickle, json
import numpy as np, pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
D = np.load(os.path.join(RES, "features.npz"))
Y, SUBJ, GOOD = D["y"], D["subj"], D["good"]
fs = sys.argv[1] if len(sys.argv) > 1 else "msk19"
B = 2000
SUBSETS = ["full", "good", "bad"]; MODES = ["raw", "dummy", "aug"]
PROTOS = ["RAND", "BLOCK", "LOSO", "GS80"]


def subset_subjects(sub):
    if sub == "full": return np.arange(36)
    return np.unique(SUBJ[GOOD == (1 if sub == "good" else 0)])


def build_runs(out, proto, which):
    """list of dict(y,p1,subj(col index)) ; which in {'sel','fin'}"""
    probs = out["probs_" + which]
    groups = {}
    for k, f in enumerate(out["folds"]):
        m = f["meta"]
        if m[0] != proto: continue
        key = 0 if proto == "BLOCK" else m[1]      # BLOCK folds pool into one out-of-fold run; LOSO pools per seed
        groups.setdefault(key, []).append(k)
    runs = []
    for key, ks in groups.items():
        idx = np.concatenate([out["folds"][k]["te"] for k in ks])
        p1 = np.concatenate([probs[k][:, 1] for k in ks])
        runs.append(dict(idx=idx, y=Y[idx], p1=p1, subjraw=SUBJ[idx]))
    return runs


def main():
    summary, boots, info_rows, subj_rows, dyn_rows, spec_rows = [], {}, [], [], [], []
    for sub in SUBSETS:
        subs = subset_subjects(sub); S = len(subs)
        smap = {s: i for i, s in enumerate(subs)}
        rng = np.random.default_rng(12345)
        W = rng.multinomial(S, np.ones(S) / S, size=B).astype(float)      # shared bootstrap resamples
        W = np.vstack([np.ones((1, S)), W])                               # row 0 = point estimate
        for mode in MODES:
            pth = os.path.join(RES, "runs", f"{fs}_{sub}_{mode}.pkl")
            if not os.path.exists(pth):
                print("missing", pth); continue
            out = pickle.load(open(pth, "rb"))
            for proto in PROTOS:
                if not any(f["meta"][0] == proto for f in out["folds"]):
                    continue
                for which in ("sel", "fin"):
                    runs = build_runs(out, proto, which)
                    for r in runs:
                        r["subj"] = np.array([smap[s] for s in r["subjraw"]])
                    res = M.boot_metrics(runs, S, W)
                    row = dict(fs=fs, subset=sub, mode=mode, proto=proto, which=which, runs=len(runs),
                               n_test=int(np.mean([len(r["y"]) for r in runs])))
                    for k, v in res.items():
                        row[k] = v[0]
                        row[k + "_lo"] = np.nanpercentile(v[1:], 2.5)
                        row[k + "_hi"] = np.nanpercentile(v[1:], 97.5)
                    # spread across repeated runs (point estimate)
                    cmr = np.stack([M.run_counts(r["y"], (r["p1"] > .5).astype(int), r["subj"], S).sum(0) for r in runs])
                    mr = M.cm_metrics(cmr)
                    row["kappa_sd_runs"] = float(np.std(mr["kappa"])); row["bacc_sd_runs"] = float(np.std(mr["bacc"]))
                    row["kappa_min_run"] = float(np.min(mr["kappa"])); row["kappa_max_run"] = float(np.max(mr["kappa"]))
                    summary.append(row)
                    boots[(sub, mode, proto, which)] = {k: v for k, v in res.items() if k in ("kappa", "bacc", "auc", "mcc")}
                    if which == "sel":
                        c = np.stack([M.run_counts(r["y"], (r["p1"] > .5).astype(int), r["subj"], S).sum(0) for r in runs]).sum(0)
                        im = M.info_metrics(c)
                        im.update(subset=sub, mode=mode, proto=proto, TP=c[0], FN=c[1], FP=c[2], TN=c[3])
                        info_rows.append(im)
                        # per-subject (recording-level) analysis for every protocol
                        for r in runs[:1] if proto not in ("LOSO",) else runs:
                            pass
                        pooled_y = np.concatenate([r["y"] for r in runs]); pooled_p = np.concatenate([r["p1"] for r in runs])
                        pooled_s = np.concatenate([r["subj"] for r in runs])
                        accs, order_ok, sbacc = [], [], []
                        for si in range(S):
                            m = pooled_s == si
                            if m.sum() == 0: continue
                            yy, pp = pooled_y[m], pooled_p[m]
                            if (yy == 0).sum() == 0 or (yy == 1).sum() == 0: continue
                            sens = ((pp > .5) & (yy == 1)).sum() / (yy == 1).sum()
                            spec = ((pp <= .5) & (yy == 0)).sum() / (yy == 0).sum()
                            sbacc.append((sens + spec) / 2)
                            order_ok.append(pp[yy == 1].mean() > pp[yy == 0].mean())
                        sbacc = np.array(sbacc); order_ok = np.array(order_ok)
                        w = st.wilcoxon(sbacc - 0.5, alternative="greater") if len(sbacc) >= 5 and np.any(sbacc != .5) else None
                        sg = st.binomtest(int(order_ok.sum()), len(order_ok), 0.5, alternative="greater")
                        subj_rows.append(dict(subset=sub, mode=mode, proto=proto, n_subj=len(sbacc),
                                              median_bacc=float(np.median(sbacc)), mean_bacc=float(sbacc.mean()),
                                              frac_bacc_gt_half=float((sbacc > .5).mean()),
                                              wilcoxon_p=float(w.pvalue) if w else np.nan,
                                              frac_ordered=float(order_ok.mean()), sign_p=float(sg.pvalue),
                                              sbacc=sbacc.tolist()))
                # training dynamics (validated protocol-level mean curves)
                idxs = [k for k, f in enumerate(out["folds"]) if f["meta"][0] == proto]
                rec = out["rec"]
                tr_l = rec["tr_loss"][idxs].mean(0); te_ce = rec["te_ce"][idxs].mean(0)
                te_acc = rec["te_acc"][idxs].mean(0); te_bacc = rec["te_bacc"][idxs].mean(0)
                tr_acc = rec["tr_acc"][idxs].mean(0)
                eb = int(np.argmin(te_ce)); ea = int(np.argmax(te_acc))
                thr = 0.95 * te_acc.max()
                dyn_rows.append(dict(subset=sub, mode=mode, proto=proto,
                                     best_acc=float(te_acc.max()), best_acc_ep=ea + 1, final_acc=float(te_acc[-1]),
                                     final_train_acc=float(tr_acc[-1]), gap_acc=float(tr_acc[-1] - te_acc[-1]),
                                     ep95=int(np.argmax(te_acc >= thr)) + 1,
                                     best_bacc=float(te_bacc.max()), final_bacc=float(te_bacc[-1]),
                                     train_ce_final=float(tr_l[-1]), test_ce_final=float(te_ce[-1]),
                                     min_test_ce=float(te_ce[eb]), min_ce_ep=eb + 1,
                                     gap_at_best=float(te_ce[eb] - tr_l[eb]), gap_final=float(te_ce[-1] - tr_l[-1]),
                                     ce_rise=float(te_ce[-1] - te_ce[eb]), p_true=float(np.exp(-te_ce[eb])),
                                     val_sel_epoch_median=float(np.median(out["best_ep"][idxs]) + 1),
                                     curve_tr=tr_l.tolist(), curve_te=te_ce.tolist()))
                # spectral entropy from final and validated weights
                for which, par in (("fin", out["par_fin"]), ("sel", out["par_sel"])):
                    vals = []
                    for k in idxs:
                        H1, e1, n1 = M.spectral(par["W1"][k].T); H2, e2, n2 = M.spectral(par["W2"][k].T)
                        g = np.abs(par["g1"][k].ravel()); pg = g / g.sum()
                        g2 = np.abs(par["g2"][k].ravel()); pg2 = g2 / g2.sum()
                        bn = 0.5 * (-(pg * np.log(pg)).sum() / np.log(len(pg)) - (pg2 * np.log(pg2)).sum() / np.log(len(pg2)))
                        vals.append([H1, e1, n1, H2, e2, n2, bn])
                    v = np.mean(vals, 0)
                    spec_rows.append(dict(subset=sub, mode=mode, proto=proto, which=which,
                                          L1_H=v[0], L1_er=v[1], L1_nER=v[2], L2_H=v[3], L2_er=v[4], L2_nER=v[5],
                                          BN_nH=v[6], mean_nER=(v[2] + v[5]) / 2))
            print("analysed", sub, mode, flush=True)
    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(RES, f"summary_{fs}.csv"), index=False)
    pd.DataFrame(info_rows).to_csv(os.path.join(RES, f"info_{fs}.csv"), index=False)
    pd.DataFrame(subj_rows).drop(columns=["sbacc"]).to_csv(os.path.join(RES, f"subject_{fs}.csv"), index=False)
    pd.DataFrame(dyn_rows).drop(columns=["curve_tr", "curve_te"]).to_csv(os.path.join(RES, f"dynamics_{fs}.csv"), index=False)
    pd.DataFrame(spec_rows).to_csv(os.path.join(RES, f"spectral_{fs}.csv"), index=False)

    # ---- interaction tests (balancing x protocol), cluster bootstrap ----
    inter = []
    for sub in SUBSETS:
        for which in ("sel", "fin"):
            for metric in ("kappa", "bacc"):
                gap = {}
                for proto in PROTOS:
                    kd = boots.get((sub, "dummy", proto, which)); ka = boots.get((sub, "aug", proto, which))
                    if kd is None or ka is None: continue
                    gap[proto] = kd[metric] - ka[metric]
                for proto, g in gap.items():
                    inter.append(dict(subset=sub, which=which, metric=metric, contrast=f"Dummy-Aug @ {proto}",
                                      est=g[0], lo=np.percentile(g[1:], 2.5), hi=np.percentile(g[1:], 97.5),
                                      p=2 * min((g[1:] >= 0).mean(), (g[1:] <= 0).mean()) if True else np.nan))
                for p1, p2 in (("RAND", "LOSO"), ("BLOCK", "LOSO"), ("RAND", "BLOCK")):
                    if p1 in gap and p2 in gap:
                        d = gap[p1] - gap[p2]
                        inter.append(dict(subset=sub, which=which, metric=metric,
                                          contrast=f"[Dummy-Aug @ {p1}] - [Dummy-Aug @ {p2}]",
                                          est=d[0], lo=np.percentile(d[1:], 2.5), hi=np.percentile(d[1:], 97.5),
                                          p=2 * min((d[1:] >= 0).mean(), (d[1:] <= 0).mean())))
    pd.DataFrame(inter).to_csv(os.path.join(RES, f"interaction_{fs}.csv"), index=False)

    # ---- protocol contrasts: leaky - honest (bootstrap) ----
    pc = []
    for sub in SUBSETS:
        for mode in MODES:
            for which in ("sel", "fin"):
                for a, b in (("RAND", "LOSO"), ("BLOCK", "LOSO"), ("RAND", "BLOCK")):
                    ka = boots.get((sub, mode, a, which)); kb = boots.get((sub, mode, b, which))
                    if ka is None or kb is None: continue
                    d = ka["kappa"] - kb["kappa"]
                    pc.append(dict(subset=sub, mode=mode, which=which, contrast=f"kappa {a} - {b}", est=d[0],
                                   lo=np.percentile(d[1:], 2.5), hi=np.percentile(d[1:], 97.5)))
    for sub in SUBSETS:
        for mode in MODES:
            for proto in PROTOS:
                ks = boots.get((sub, mode, proto, "sel")); kf = boots.get((sub, mode, proto, "fin"))
                if ks is None or kf is None: continue
                d = kf["kappa"] - ks["kappa"]
                pc.append(dict(subset=sub, mode=mode, which="fin-sel", contrast=f"kappa final - selected @ {proto}", est=d[0],
                               lo=np.percentile(d[1:], 2.5), hi=np.percentile(d[1:], 97.5)))
    pd.DataFrame(pc).to_csv(os.path.join(RES, f"protocol_contrast_{fs}.csv"), index=False)
    pickle.dump(dict(subj_rows=subj_rows, dyn_rows=dyn_rows), open(os.path.join(RES, f"extra_{fs}.pkl"), "wb"))
    print("saved")


if __name__ == "__main__":
    main()
