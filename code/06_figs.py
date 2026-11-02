"""Step 6: figures for the manuscript."""
import os, pickle, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
fs = "msk19"
S = pd.read_csv(os.path.join(RES, f"summary_{fs}.csv"))
I = pd.read_csv(os.path.join(RES, f"info_{fs}.csv"))
ex = pickle.load(open(os.path.join(RES, f"extra_{fs}.pkl"), "rb"))
SUBS = ["full", "good", "bad"]; MODES = ["raw", "dummy", "aug"]
NAME = dict(full="Full", good="Good Count", bad="Bad Count", raw="raw", dummy="+ Dummy", aug="+ Aug")
PC = dict(RAND="#d95f02", BLOCK="#e6ab02", LOSO="#1b9e77", GS80="#7570b3")
PN = dict(RAND="RAND (random epochs)", BLOCK="BLOCK (within-subject, blocked)", LOSO="LOSO (subject-independent)", GS80="GS80 (grouped 80/20)")
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})


def cm_fig(proto, fname, title):
    fig, axs = plt.subplots(3, 3, figsize=(6.6, 6.2))
    for i, sub in enumerate(SUBS):
        for j, mode in enumerate(MODES):
            r = I[(I.subset == sub) & (I["mode"] == mode) & (I.proto == proto)].iloc[0]
            c = np.array([[r.TN, r.FP], [r.FN, r.TP]])
            cn = c / c.sum(1, keepdims=True)
            ax = axs[i, j]
            ax.imshow(cn, cmap="Blues", vmin=0, vmax=1)
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{int(c[a, b])}\n({cn[a, b]*100:.0f}%)", ha="center", va="center",
                            color="white" if cn[a, b] > .55 else "black", fontsize=8)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["rest", "stress"]); ax.set_yticklabels(["rest", "stress"])
            ax.set_title(f"{NAME[sub]} {NAME[mode]}", fontsize=8.5)
            if j == 0: ax.set_ylabel("true")
            if i == 2: ax.set_xlabel("predicted")
    fig.suptitle(title, fontsize=9.5)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(RES, fname), dpi=250); plt.close(fig)


cm_fig("RAND", "fig2_cm_rand.png", "Pooled confusion matrices, RAND protocol (validation-selected models, 10 repeats)")
cm_fig("LOSO", "fig3_cm_loso.png", "Pooled confusion matrices, LOSO protocol (validation-selected models, 3 seeds)")

# kappa forest
fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True)
cells = [(s, m) for s in SUBS for m in MODES]
for ax, which, ttl in zip(axs, ["sel", "fin"], ["validation-selected model", "final-epoch (1000) model"]):
    for k, p in enumerate(["RAND", "BLOCK", "LOSO", "GS80"]):
        ys, xs, lo, hi = [], [], [], []
        for ci, (s, m) in enumerate(cells):
            r = S[(S.subset == s) & (S["mode"] == m) & (S.proto == p) & (S.which == which)].iloc[0]
            ys.append(ci + (k - 1.5) * 0.17); xs.append(r.kappa); lo.append(r.kappa - r.kappa_lo); hi.append(r.kappa_hi - r.kappa)
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", ms=3.5, color=PC[p], lw=1, label=PN[p] if which == "sel" else None)
    ax.axvline(0, color="grey", lw=.6)
    ax.set_title(ttl, fontsize=9); ax.set_xlabel("Cohen's kappa (95% CI)")
axs[0].set_yticks(range(9)); axs[0].set_yticklabels([f"{NAME[s]} {NAME[m]}" for s, m in cells]); axs[0].invert_yaxis()
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7, loc="upper center", ncol=2, frameon=False)
plt.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(RES, "fig4_kappa_forest.png"), dpi=250); plt.close(fig)

# per-subject balanced accuracy RAND vs LOSO (validated)
sr = pd.DataFrame(ex["subj_rows"])
fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
for ax, sub in zip(axs, SUBS):
    data, pos, cols = [], [], []
    for j, mode in enumerate(MODES):
        for k, p in enumerate(["RAND", "LOSO"]):
            row = sr[(sr.subset == sub) & (sr["mode"] == mode) & (sr.proto == p)].iloc[0]
            data.append(row.sbacc); pos.append(j * 3 + k); cols.append(PC[p])
    bp = ax.boxplot(data, positions=pos, widths=.8, patch_artist=True, showfliers=True, flierprops=dict(ms=2))
    for patch, c in zip(bp["boxes"], cols): patch.set_facecolor(c); patch.set_alpha(.7)
    for med in bp["medians"]: med.set_color("k")
    ax.axhline(.5, color="grey", lw=.6, ls="--")
    ax.set_xticks([.5, 3.5, 6.5]); ax.set_xticklabels(["raw", "Dummy", "Aug"]); ax.set_title(NAME[sub], fontsize=9)
axs[0].set_ylabel("per-subject balanced accuracy")
axs[0].plot([], [], "s", color=PC["RAND"], label="RAND"); axs[0].plot([], [], "s", color=PC["LOSO"], label="LOSO")
axs[0].legend(fontsize=7, frameon=False, loc="lower right")
plt.tight_layout(); fig.savefig(os.path.join(RES, "fig5_subject_bacc.png"), dpi=250); plt.close(fig)

# cross-entropy curves
dyn = pd.DataFrame(ex["dyn_rows"])
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
for ax, p in zip(axs, ["RAND", "BLOCK", "LOSO"]):
    for mode, ls in zip(MODES, ["-", "--", ":"]):
        r = dyn[(dyn.subset == "full") & (dyn["mode"] == mode) & (dyn.proto == p)].iloc[0]
        x = np.arange(1, 1001)
        ax.plot(x, r.curve_te, ls, color="#c0392b", lw=1.2, label=f"test CE, {mode}")
        ax.plot(x, r.curve_tr, ls, color="#2c6fbb", lw=1.2, label=f"train loss, {mode}")
    ax.set_xscale("log"); ax.set_title(p + " (Full)", fontsize=9); ax.set_xlabel("epoch")
axs[0].set_ylabel("cross-entropy (nats)"); axs[0].legend(fontsize=6, frameon=False, ncol=2)
plt.tight_layout(); fig.savefig(os.path.join(RES, "fig6_ce_curves.png"), dpi=250); plt.close(fig)
print("figs done")
