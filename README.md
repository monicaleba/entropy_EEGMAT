# Code: subject-dependent vs subject-independent evaluation on EEGMAT

Analysis code for the revised manuscript *When 99% Does Not Transfer: Subject-Dependent versus
Subject-Independent Evaluation of EEG Mental-Stress Classification on EEGMAT*.
It replaces the earlier code in `src/`.

## Setup
```
pip install -r requirements.txt
```
Download **EEG During Mental Arithmetic Tasks** (EEGMAT) from PhysioNet
(<https://physionet.org/content/eegmat/1.0.0/>) and place the 72 `.edf` files and
`subject-info.csv` in a folder `eeg-mental/` **next to** this `code/` folder:

```
<repo root>/
├── code/            (this folder)
├── eeg-mental/      (EEGMAT files, not included)
└── results/         (created by the scripts, not versioned)
```

## Run order
| Step | Command | Output |
|---|---|---|
| 1 | `python 01_features.py` | epoch features (mean, SD, skewness, kurtosis, RMS) and augmentation bank |
| 2 | `python 02_run.py msk19 1000 RAND,BLOCK,LOSO,GS80 8` | trained networks, 3 pools x 3 training-set modes x 4 protocols (main grid) |
| 2b | `python 02b_pre.py` | control: balance first, then split (original order of operations) |
| 2c | `python 02_run.py msr19,msr21 1000 RAND,LOSO 5` | feature-set ablation (RMS, 19 / 21 channels) |
| 3 | `python 03_analyze.py msk19` (also `msr19`, `msr21`) | tables with subject-cluster bootstrap CIs, interaction tests, information metrics, dynamics, spectral entropy |
| 3b | `python 03b_pre_analyze.py` | control analysis |
| 4 | `python 04_shap.py msk19` | SHAP explanation entropy |
| 5 | `python 05_fig_workflow.py`, `python 06_figs.py` | figures |

Arguments of `02_run.py`: feature set, epochs, comma-separated protocols, number of parallel processes.
The full grid trains more than 1,000 MLPs in parallel on CPU (about 25-40 min on 24 cores).

## Files
- `engine.py`: batched MLP trainer (57-64-32-2, Adam 0.005, batch 32, validation-selected and final-epoch models).
- `metrics.py`: confusion-matrix metrics, subject-cluster bootstrap, mutual information, NMI, AMI, confusion entropy, spectral entropy.
- Protocols: `RAND` random epoch split; `BLOCK` within-subject chronologically blocked; `LOSO` leave-one-subject-out; `GS80` repeated grouped 80/20.
- Balancing (Dummy duplication, noise/scale/shift augmentation) is applied inside training partitions only.

Seeds are fixed throughout. Results may differ in the last digits across hardware or library versions.
