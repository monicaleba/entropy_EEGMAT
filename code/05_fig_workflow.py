"""Figure 1: map of the experimental design (requested by Reviewer 1)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fig, ax = plt.subplots(figsize=(7.4, 8.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 118); ax.axis("off")

C = dict(data="#dbe9f6", proc="#e6f0dc", proto="#fde9d3", train="#efe3f3", ev="#f9f3c8", edge="#444444")


def box(x, y, w, h, text, fc, fs=8.2, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=C["edge"], lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            linespacing=1.25)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.0, color=C["edge"]))


# 1 data
box(10, 105, 80, 10, "EEGMAT: 36 subjects  x  2 recordings (rest ~182 s, task ~62 s), 500 Hz\n"
    "26 Good counters, 10 Bad counters;  label = recording (rest = 0, task = 1)", C["data"], bold=False)
arrow(50, 105, 50, 100)
box(10, 90, 80, 10, "Preprocessing: 19 scalp EEG channels, band-pass 0.5-45 Hz, 4-s non-overlapping epochs\n"
    "2134 epochs (1594 rest, 540 task)   |   features per channel: mean, SD, skewness (57-D)", C["proc"])
arrow(50, 90, 50, 85)
box(10, 76, 80, 9, "Three epoch pools (subsets):   Full (36 subj.)     Good (26)     Bad (10)", C["data"])
arrow(50, 76, 50, 71)

# 2 protocols
ax.text(50, 69.5, "Four evaluation protocols (same pools, same model, same budget)", ha="center", fontsize=8.6, fontweight="bold")
px = [2, 26, 50, 74]
txt = ["RAND\nrandom epoch split\n80/20, 10 repeats\n(subject-dependent,\nliterature-style)",
       "BLOCK\nwithin-subject,\nchronological blocks\n5-fold, 1-epoch purge\n(subject-dependent)",
       "LOSO\nleave-one-subject-out\n3 seeds\n(subject-independent,\nmain protocol)",
       "GS80\nrepeated grouped 80/20\n20 repeats\n(subject-independent,\noriginal design)"]
for x, t in zip(px, txt):
    box(x, 52, 23.5, 15, t, C["proto"], fs=7.6)
    arrow(x + 11.75, 52, 50, 47) if False else None
for x in px:
    arrow(x + 11.75, 52, x + 11.75, 47.5)
ax.plot([px[0] + 11.75, px[3] + 11.75], [47.5, 47.5], color=C["edge"], lw=1.0)
arrow(50, 47.5, 50, 44.5)

# 3 training
box(6, 29, 88, 15.5, "Inside every training partition ONLY (validation and test stay untouched, natural class ratio)\n"
    "raw   |   Dummy: random duplication of minority epochs   |   Aug: noise + scale + shift copies\n"
    "z-scoring fitted on real training epochs;  validation part (15%) held out for model selection\n"
    "MLP 57-64-32-2, Adam 0.005, batch 32, 1000 epochs   ->   report final-epoch AND validation-selected model",
    C["train"], fs=8.0)
arrow(50, 29, 50, 25.5)

# 4 evaluation
box(2, 3, 29, 22, "Performance (test only)\nAcc, balanced Acc, kappa,\nMCC, ROC-AUC, F1\n95% CI: subject-cluster\nbootstrap (2000)", C["ev"], fs=7.8)
box(35, 3, 29, 22, "Inference\nper-subject balanced Acc,\nsign / Wilcoxon tests,\nbalancing x protocol\ninteraction (bootstrap)", C["ev"], fs=7.8)
box(68, 3, 30, 22, "Information & internals\nH(Y), I(Y;Yhat), NMI, AMI, CEN,\ncross-entropy dynamics,\nweight spectral entropy,\nSHAP explanation entropy", C["ev"], fs=7.8)
for x in (16.5, 49.5, 83):
    arrow(50, 25.5, x, 25.0) if False else None
plt.tight_layout(pad=0.3)
fig.savefig(os.path.join(ROOT, "results", "fig1_workflow.png"), dpi=300)
print("ok")
