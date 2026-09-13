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

## Results at a glance

| | data-leak | non-leak |
|---|---|---|
| Accuracy | 88.2–99.2% (mean 94.0%) | 43.8–85.0% (mean 74.3%) |
| Cohen's κ | 0.664–0.977 | −0.32–0.667 |
| Best model | Bad-Count + dummy (κ 0.977) | Bad-Count + dummy (κ 0.667, AUC 0.996) |
| Failure mode | none | 3 unbalanced regimes collapse to κ ≤ 0.13 |

## Citation

If you use this code or the accompanying results, please cite the paper 

> Sibisanu, R., Leba, M., Ionica, A. (2026). When 99% Isn't Real: Quantifying Subject-Identity Leakage
> in EEGMAT Stress Classification. *Submitted to Entropy*.

and the dataset:

> Zyma, I., Tukaev, S., Seleznov, I., Kiyono, K., Popov, A., Chernykh, M., & Shpenkov, O.
> (2019). Electroencephalograms during mental arithmetic task performance. *Data*, 4(1), 14.

## License

Code is MIT-licensed (see [LICENSE](LICENSE)). The EEGMAT dataset is not redistributed here and
is subject to its own PhysioNet license terms.
