# When 99% Isn't Real: Quantifying Subject-Identity Leakage in EEGMAT Stress Classification

Code, trained-model artifacts, and reproducibility reports for a study of **subject-level
data leakage** in EEG relax-vs-stress classification on the public
[EEGMAT (PhysioNet) mental-arithmetic dataset](https://physionet.org/content/eegmat/1.0.0/).

**Short abstract.** Reported accuracies for EEG-based relax-versus-stress classification are
often inflated by subject-level data leakage: when epochs from the same participant appear on
both sides of a train/test split, the classifier exploits subject identity as an unintended
information channel. Using EEGMAT and a multilayer perceptron trained on 63 time-domain
features (mean, SD, RMS across 21 channels), we contrast a leakage-permitting epoch-level split
with a strict subject-grouped split, crossed with seven dataset-construction modes (full pool;
Good- and Bad-Counter subsets, each raw, duplicate-balanced, or augmented) — 14 models in total.
With leakage, accuracy reaches 88.2–99.2% (κ 0.664–0.977), matching published windowed-split
results. Under subject-grouped evaluation, unbalanced regimes collapse to chance or below
(κ as low as −0.32); only balanced regimes retain genuine signal, the best reaching 85.0%
accuracy, κ 0.667, ROC-AUC 0.996. Confusion-matrix mutual information shows leaky models resolve
36–93% of label entropy versus only 2–53% without leakage, but is sign-blind — two below-chance
non-leak models still register positive mutual information — so it must be interpreted alongside
κ and AUC. Cross-entropy training-loss divergence (up to 3.22 nats) offers a complementary
leakage diagnostic, while weight spectral entropy and SHAP explanation entropy stay invariant
across regimes, showing leakage changes what is learned, not model capacity. We argue for
leakage-aware, subject-independent evaluation paired with explicit information-theoretic
diagnostics.

## Repository layout

```
├── src/
│   ├── leak/                 # DATA-LEAK condition: epoch-level train/test split
│   │   ├── leak_mlpBF.py     #   full 36-subject pool
│   │   ├── leak_mlpBGC.py    #   Good-Counter subjects only
│   │   └── leak_mlpBBC.py    #   Bad-Counter subjects only
│   ├── no_leak_shap/         # NON-LEAK condition: subject-grouped split + SHAP
│   │   ├── mshap_mlpBF.py
│   │   ├── mshap_mlpBGC.py
│   │   └── mshap_mlpBCC.py   #   (Bad-Counter; see naming note below)
│   └── augment_eegmat.py     # leakage-safe signal augmentation for class balancing
├── analysis/                 # post-hoc entropy / performance metrics + report builders
│   ├── entropy_analysis.py   #   -> ./out/*.csv, *.json   (run first)
│   ├── make_figs.py          #   -> ./out/figs/*.png      (run second)
│   ├── build_report.py       #   -> ../reports/entropy_report.html
│   ├── build_docx.py         #   -> ../reports/EEGMAT_MLP_entropy_metrics.docx
│   ├── perf_analysis.py      #   -> ./out_perf/*.csv, figs/*.png
│   ├── build_perf_docx.py    #   -> ../reports/EEGMAT_MLP_performance_report.docx
│   └── entropy_mi_compute.py #   standalone MI/entropy check -- see note below
├── results/                  # small artifacts produced by the 14 training runs
│   ├── data-leak/             {data-leak.txt, results_*/{model.pth, history.npy,
│   │                           confusion_matrix.png, metrics_per_epoch.png}}
│   └── non-leak/               same, plus shap_outputs_*/ (SHAP CSVs + figures)
├── reports/                  # final generated deliverables (HTML + 2 Word reports)
├── requirements.txt
└── LICENSE
```

The raw EEGMAT `.edf` recordings are **not** included (size and PhysioNet licensing) — see
[Dataset](#dataset) below.

## Dataset

Download **EEG During Mental Arithmetic Tasks** from PhysioNet:
<https://physionet.org/content/eegmat/1.0.0/>, then arrange the recordings by relax/stress
label and subject subset:

```
data_BF/relaxed/*.edf         data_BF/non-relaxed/*.edf          # full 36-subject pool
data_BGC/relaxed/*.edf        data_BGC/non-relaxed/*.edf         # Good-Counter subjects (~26)
data_BBC/relaxed/*.edf        data_BBC/non-relaxed/*.edf         # Bad-Counter subjects  (~10)
                                                                  # (data_BCC for the no_leak_shap scripts)
```

Good/Bad-Counter status is defined by Zyma et al. (2019) from each subject's subtraction-task
performance (26 good counters, 10 bad counters). "relaxed" = the pre-task baseline recording;
"non-relaxed" = the during-task mental-arithmetic recording.

**Balancing.** For the `+dummy` regimes referenced in the paper, minority-class files were
duplicated (see the caveat this repository raises about that approach in `src/augment_eegmat.py`'s
docstring). For the `+augmented` regimes, run `src/augment_eegmat.py` instead — it generates
noise/scale/shift-perturbed copies and *encodes the source subject in the filename*, so a
group-aware split (used by every `no_leak_shap` script) never separates a subject's real and
augmented recordings across train/test:

```bash
python src/augment_eegmat.py --data_dir data_BBC --minority_class non-relaxed --balance
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Python ≥3.10 recommended (developed/tested on 3.10). GPU is not required — all models are a
small 63→64→32→2 MLP trained on tabular features, not raw EEG.

## Usage

### 1. Train + evaluate a model

Each script is self-contained; run it from the directory containing its expected `data_*/`
folder (or edit `DATA_DIR`/`RESULTS_DIR` at the top of the file):

| Script | Condition | Subjects | Output |
|---|---|---|---|
| `src/leak/leak_mlpBF.py` | data-leak | all 36 | `results_BF/` |
| `src/leak/leak_mlpBGC.py` | data-leak | Good-Counter | `results_BGC/` |
| `src/leak/leak_mlpBBC.py` | data-leak | Bad-Counter | `results_BBC/` |
| `src/no_leak_shap/mshap_mlpBF.py` | non-leak + SHAP | all 36 | `results_BF/`, `shap_outputs_BF/` |
| `src/no_leak_shap/mshap_mlpBGC.py` | non-leak + SHAP | Good-Counter | `results_BGC/`, `shap_outputs_BGC/` |
| `src/no_leak_shap/mshap_mlpBCC.py` | non-leak + SHAP | Bad-Counter | `results_BCC/`, `shap_outputs_BCC/` |

```bash
cd src/leak && python leak_mlpBF.py
```

Each run trains for 1000 epochs, then writes `model.pth`, `history.npy` (full per-epoch loss /
accuracy / precision / recall / F1 / κ), `confusion_matrix.png`, and `metrics_per_epoch.png`.
The `no_leak_shap` scripts additionally run a SHAP explainability pass and write
`shap_beeswarm.png`, `shap_feature_importance.{png,csv}`, `shap_channel_importance.{png,csv}`,
and `shap_values_raw.csv`.

To reproduce the "+dummy" and "+augmented" regimes, run the same script against a `data_*`
folder that has been balanced by duplication or by `augment_eegmat.py` respectively, pointing
`RESULTS_DIR`/`SHAP_OUTPUT_DIR` at a distinct name (e.g. `results_BGCD`, `results_BGCA`).

### 2. Reproduce the entropy / performance reports

The `results/` folder already contains everything these scripts need (no raw EEG required):

```bash
cd analysis
python entropy_analysis.py   # confusion-matrix info theory, CE dynamics, weight/SHAP entropy
python make_figs.py          # 7 figures from the above
python build_report.py       # -> ../reports/entropy_report.html
python build_docx.py         # -> ../reports/EEGMAT_MLP_entropy_metrics.docx

python perf_analysis.py      # accuracy / kappa / MCC / AUC / per-class / training-dynamics tables
python build_perf_docx.py    # -> ../reports/EEGMAT_MLP_performance_report.docx
```

`analysis/entropy_mi_compute.py` is a separate, standalone confusion-matrix MI/entropy
calculator kept for cross-checking (see note below) — it takes no input files, just run it.

## Results at a glance

| | data-leak | non-leak |
|---|---|---|
| Accuracy | 88.2–99.2% (mean 94.0%) | 43.8–85.0% (mean 74.3%) |
| Cohen's κ | 0.664–0.977 | −0.32–0.667 |
| Best model | Bad-Count + dummy (κ 0.977) | Bad-Count + dummy (κ 0.667, AUC 0.996) |
| Failure mode | none | 3 unbalanced regimes collapse to κ ≤ 0.13 |

Full tables, figures, and the EEGMAT-literature comparison are in
[`reports/EEGMAT_MLP_performance_report.docx`](reports/EEGMAT_MLP_performance_report.docx);
the information-theoretic breakdown (mutual information, cross-entropy dynamics, weight/SHAP
entropy) is in
[`reports/EEGMAT_MLP_entropy_metrics.docx`](reports/EEGMAT_MLP_entropy_metrics.docx) /
[`reports/entropy_report.html`](reports/entropy_report.html).

## Notes on the provided scripts

A few things worth knowing before you run or extend these scripts:

- **`mshap_mlpBF.py` had three bugs in the version originally added to this repo** that would
  have prevented it from running at all (an unindented `print(...)` after an `if`, a stray
  `.save(...)` missing its `np` receiver, and a missing `torch.save(model.state_dict(), ...)`
  call that silently would have dropped the model checkpoint). All three are fixed here — see
  the docstring at the top of that file for the exact diff. A cosmetic console-banner label
  ("BAD COUNT", copy-pasted into the Good-Count and Full-dataset scripts) was also corrected in
  `mshap_mlpBGC.py` and `mshap_mlpBF.py`, and the three `mshap_mlp*.py` scripts' per-run summary
  CSVs were all writing to the *same* hardcoded filename (`mlpBBC_metrics_summary.csv`) —
  changed to write `metrics_summary.csv` / `classification_report.csv` inside each script's own
  `RESULTS_DIR` so three runs from one working directory no longer overwrite each other.
- **Bad-Counter naming is inconsistent between the two conditions**: the data-leak script is
  `leak_mlpBBC.py` / `data_BBC`, while its non-leak counterpart is `mshap_mlpBCC.py` /
  `data_BCC`. Both refer to the same 10-subject Bad-Counter group — this is carried over from
  the original scripts rather than introduced here; renamed folders will need matching
  `DATA_DIR`/`RESULTS_DIR` edits.
- **`analysis/entropy_mi_compute.py` encodes a different set of confusion matrices** than
  `results/data-leak/data-leak.txt` and `results/non-leak/non-leak.txt` (e.g. its "Full Dataset"
  leaky matrix is `[[81,15],[14,270]]` vs. `[[64,32],[13,271]]` reconstructed from the shipped
  `data-leak.txt`), and references additional tables/feature sets ("TBA", "Time+TBA") not
  otherwise present in this repository. This looks like a snapshot from a different or later
  experiment batch than the 14 runs under `results/` — it is kept here as-is for reference, but
  its numbers should **not** be assumed to match `reports/EEGMAT_MLP_entropy_metrics.docx`
  without checking which experiment batch it corresponds to.

## Citation

If you use this code or the accompanying results, please cite the paper (details to be added
once published) and the dataset:

> Zyma, I., Tukaev, S., Seleznov, I., Kiyono, K., Popov, A., Chernykh, M., & Shpenkov, O.
> (2019). Electroencephalograms during mental arithmetic task performance. *Data*, 4(1), 14.

`src/augment_eegmat.py`'s docstring also references the duplicate-balancing approach it argues
against:

> Salankar, N., Koundal, D., & Qaisar, S. M. (2021). Stress classification by multimodal
> physiological signals using variational mode decomposition and machine learning. *Journal of
> Healthcare Engineering*, 2021, Article 2146369.

Full reference lists for every entropy/information-theoretic metric used are in
[`reports/EEGMAT_MLP_entropy_metrics.docx`](reports/EEGMAT_MLP_entropy_metrics.docx) and the
EEGMAT-literature comparison table is in
[`reports/EEGMAT_MLP_performance_report.docx`](reports/EEGMAT_MLP_performance_report.docx).

## License

Code is MIT-licensed (see [LICENSE](LICENSE)). The EEGMAT dataset is not redistributed here and
is subject to its own PhysioNet license terms.
