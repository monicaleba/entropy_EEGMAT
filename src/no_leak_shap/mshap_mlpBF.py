"""
mshap_mlpBF.py -- FULL DATASET (all 36 EEGMAT subjects), relaxed vs. one
mental-arithmetic stress state.

Data-handling condition: NON-LEAK -- see mshap_mlpBCC.py for details.
Same pipeline as mshap_mlpBCC.py, pointed at the full 36-subject pool.

Expects ./data_BF/relaxed/*.edf and ./data_BF/non-relaxed/*.edf.
Outputs: ./results_BF/{model.pth, history.npy, confusion_matrix.png,
metrics_per_epoch.png} and ./shap_outputs_BF/{shap_beeswarm.png,
shap_feature_importance.{png,csv}, shap_channel_importance.{png,csv},
shap_values_raw.csv}.

FIXES APPLIED vs. the original file (which would not run as-is):
  1. An `if (epoch + 1) % 5 == 0:` block had an unindented `print(...)`
     immediately after it -- a Python IndentationError. Re-indented.
  2. The history was saved with a stray `   .save(...)` (missing the `np`
     receiver) -- another SyntaxError. Restored to `np.save(...)`.
  3. `torch.save(model.state_dict(), ...)` was missing entirely, so no
     model.pth would have been written. Added, matching mshap_mlpBCC.py.
  4. The console banner before the final metrics printed "BAD COUNT"
     (copy-pasted from mshap_mlpBCC.py) -- corrected to "FULL DATASET".
None of these change the model, features, split, or any computed metric;
they only make the script runnable and its console/file output correct.
"""
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import seaborn as sns
import mne
import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display required
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
    precision_recall_fscore_support,
)


CHANNELS = [
    "EEG Fp1", "EEG Fp2", "EEG F3", "EEG F4", "EEG F7", "EEG F8",
    "EEG T3", "EEG T4", "EEG C3", "EEG C4", "EEG T5", "EEG T6",
    "EEG P3", "EEG P4", "EEG O1", "EEG O2",
    "EEG Fz", "EEG Cz", "EEG Pz",
    "EEG A2-A1", "ECG ECG"
]

SFREQ = 500.0
WINDOW_DURATION = 4.0
EPOCH_LENGTH = int(SFREQ * WINDOW_DURATION)

DATA_DIR = "data_BF"
RANDOM_STATE = 101
TEST_SIZE = 0.2
BATCH_SIZE = 16
NUM_EPOCHS = 1000
RESULTS_DIR = "results_BF"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# SHAP settings
# ---------------------------------------------------------------------
SHAP_OUTPUT_DIR = "shap_outputs_BF"
SHAP_BACKGROUND_SIZE = 100   # training rows used as SHAP background reference
SHAP_TEST_SAMPLE_SIZE = 200  # test rows to explain (cost scales with this)


def load_and_preprocess_edf(file_path):
    raw = mne.io.read_raw_edf(
        file_path,
        preload=True,
        verbose=True
    )

    try:
        raw.pick(CHANNELS)
    except ValueError as e:
        print(f"Eroare canale în {file_path}: {e}")
        return None

    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose=False)

    raw.filter(
        l_freq=0.5,
        h_freq=45.0,
        verbose=False
    )

    data, _ = raw[:, :]

    total_samples = data.shape[1]
    num_epochs = total_samples // EPOCH_LENGTH

    epochs = []

    for i in range(num_epochs):
        start = i * EPOCH_LENGTH
        end = start + EPOCH_LENGTH

        epochs.append(data[:, start:end])

    return np.array(epochs)


def load_dataset_from_folders(data_dir):
    X = []
    y = []
    groups = []

    class_names = sorted(
        [
            directory
            for directory in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, directory))
        ]
    )

    class_to_idx = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    print("Clase detectate:", class_to_idx)

    # Every EDF file receives a unique group ID
    group_id = 0

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        label = class_to_idx[class_name]

        for filename in os.listdir(class_dir):
            if not filename.lower().endswith(".edf"):
                continue

            file_path = os.path.join(class_dir, filename)

            print(f"Se procesează: {file_path}")

            epochs = load_and_preprocess_edf(file_path)

            if epochs is None or len(epochs) == 0:
                continue

            # Add all epochs from this EDF file
            X.append(epochs)

            # Same label for all epochs from this EDF file
            y.extend([label] * len(epochs))

            # Same group ID for all epochs from this EDF file
            groups.extend([group_id] * len(epochs))

            group_id += 1

    if len(X) == 0:
        return None, None, None, class_to_idx

    X = np.concatenate(X, axis=0)
    y = np.array(y)
    groups = np.array(groups)

    return X, y, groups, class_to_idx


def extract_features(epochs_data):
    features = []

    for epoch in epochs_data:
        epoch_features = []

        for channel_data in epoch:
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            rms = np.sqrt(np.mean(channel_data ** 2))

            epoch_features.extend([
                mean,
                std,
                rms
            ])

        features.append(epoch_features)

    return np.array(features)


def build_feature_names(channels):
    """
    Reproduce the exact feature ordering used by extract_features(): for
    each channel, [mean, std, rms].
    """
    stats = ("mean", "std", "rms")
    return [f"{ch}_{stat}" for ch in channels for stat in stats]

@torch.no_grad()
def evaluate(model, X_tensor, y_tensor):
    model.eval()
    outputs = model(X_tensor)
    _, predictions = torch.max(outputs, 1)

    y_true = y_tensor.cpu().numpy()
    y_pred = predictions.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    loss = nn.functional.cross_entropy(outputs, y_tensor).item()

    return {
        "loss": loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "kappa": kappa,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_metrics_history(history, results_dir=RESULTS_DIR):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0, 0].plot(epochs, history["test_loss"], label="Test Loss")
    axes[0, 0].set_title("Loss per Epoch")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[0, 1].plot(epochs, history["test_acc"], label="Test Acc")
    axes[0, 1].set_title("Accuracy per Epoch")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(epochs, history["test_f1"], label="Test F1 (weighted)")
    axes[1, 0].plot(epochs, history["test_precision"], label="Test Precision")
    axes[1, 0].plot(epochs, history["test_recall"], label="Test Recall")
    axes[1, 0].set_title("Precision / Recall / F1 per Epoch")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(epochs, history["test_kappa"], label="Test Cohen's κ")
    axes[1, 1].set_title("Cohen's Kappa per Epoch")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    fig.tight_layout()
    path = os.path.join(results_dir, "metrics_per_epoch.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Grafic metrici salvat: {path}")


def plot_confusion_matrix_chart(y_true, y_pred, target_names, results_dir=RESULTS_DIR):
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names,
                ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names,
                ax=axes[1])
    axes[1].set_title("Confusion Matrix (normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    fig.tight_layout()
    path = os.path.join(results_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Grafic matrice de confuzie salvat: {path}")

class EEGMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------
def make_predict_fn(model: nn.Module):
    model.eval()

    def predict_fn(x_numpy: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    return predict_fn


def run_shap_analysis(
    model,
    X_train_scaled,
    X_test_scaled,
    feature_names,
    class_names,
    output_dir=SHAP_OUTPUT_DIR,
    background_size=SHAP_BACKGROUND_SIZE,
    test_sample_size=SHAP_TEST_SAMPLE_SIZE,
    random_state=RANDOM_STATE,
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(random_state)

    assert X_train_scaled.shape[1] == len(feature_names), (
        f"feature_names has {len(feature_names)} entries but "
        f"X_train_scaled has {X_train_scaled.shape[1]} columns -- "
        f"these must match exactly."
    )

    if X_train_scaled.shape[0] > background_size:
        bg_idx = rng.choice(X_train_scaled.shape[0], size=background_size, replace=False)
        background = X_train_scaled[bg_idx]
    else:
        background = X_train_scaled

    if X_test_scaled.shape[0] > test_sample_size:
        test_idx = rng.choice(X_test_scaled.shape[0], size=test_sample_size, replace=False)
        X_explain = X_test_scaled[test_idx]
    else:
        X_explain = X_test_scaled

    predict_fn = make_predict_fn(model)

    print(f"\nBuilding SHAP explainer (background={background.shape[0]} rows, "
          f"explaining {X_explain.shape[0]} test rows)...")

    explainer = shap.Explainer(predict_fn, background, feature_names=feature_names)
    shap_explanation = explainer(X_explain)

    class_index = 1 if len(class_names) == 2 else None

    if class_index is not None and shap_explanation.values.ndim == 3:
        shap_values_for_class = shap_explanation.values[:, :, class_index]
        class_label = class_names[class_index]
    else:
        shap_values_for_class = shap_explanation.values
        class_label = class_names[0] if class_names else "target"

    # --- Beeswarm summary plot ---
    plt.figure()
    shap.summary_plot(shap_values_for_class, X_explain, feature_names=feature_names, show=False)
    plt.title(f"SHAP summary — predicting '{class_label}'")
    plt.tight_layout()
    beeswarm_path = os.path.join(output_dir, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=200)
    plt.close()
    print(f"Saved: {beeswarm_path}")

    # --- Global feature importance (mean |SHAP|) ---
    mean_abs_shap = np.abs(shap_values_for_class).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    top_n = min(20, len(feature_names))
    plt.figure(figsize=(8, max(4, top_n * 0.3)))
    top_features = importance_df.head(top_n).iloc[::-1]
    plt.barh(top_features["feature"], top_features["mean_abs_shap"])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top {top_n} features — predicting '{class_label}'")
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "shap_feature_importance.png")
    plt.savefig(bar_path, dpi=200)
    plt.close()
    print(f"Saved: {bar_path}")

    # --- Channel-level aggregation (sum |SHAP| across mean/std/rms) ---
    channel_names = sorted(
        set(name.rsplit("_", 1)[0] for name in feature_names),
        key=lambda ch: feature_names.index(f"{ch}_mean"),
    )

    channel_importance = {}
    for ch in channel_names:
        stat_cols = [i for i, name in enumerate(feature_names) if name.startswith(f"{ch}_")]
        channel_importance[ch] = mean_abs_shap[stat_cols].sum()

    channel_df = pd.DataFrame({
        "channel": list(channel_importance.keys()),
        "summed_abs_shap": list(channel_importance.values()),
    }).sort_values("summed_abs_shap", ascending=False)

    plt.figure(figsize=(8, max(4, len(channel_df) * 0.3)))
    plot_df = channel_df.iloc[::-1]
    plt.barh(plot_df["channel"], plot_df["summed_abs_shap"])
    plt.xlabel("Summed |SHAP value| across mean/std/rms")
    plt.title(f"Channel-level importance — predicting '{class_label}'")
    plt.tight_layout()
    channel_path = os.path.join(output_dir, "shap_channel_importance.png")
    plt.savefig(channel_path, dpi=200)
    plt.close()
    print(f"Saved: {channel_path}")

    # --- Raw values + importance tables, as CSV ---
    pd.DataFrame(shap_values_for_class, columns=feature_names).to_csv(
        os.path.join(output_dir, "shap_values_raw.csv"), index=False
    )
    importance_df.to_csv(os.path.join(output_dir, "shap_feature_importance.csv"), index=False)
    channel_df.to_csv(os.path.join(output_dir, "shap_channel_importance.csv"), index=False)

    print(f"\nTop 5 features by mean |SHAP|:\n{importance_df.head(5).to_string(index=False)}")
    print(f"\nTop 5 channels by summed |SHAP|:\n{channel_df.head(5).to_string(index=False)}")

    return importance_df, channel_df


if __name__ == "__main__":

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------
    epochs_data, y_labels, groups, class_to_idx = (
        load_dataset_from_folders(DATA_DIR)
    )

    if epochs_data is None:
        print("Nu s-au găsit fișiere EDF.")
        raise SystemExit

    print(f"\nEpoci încărcate: {epochs_data.shape}")
    print(f"Număr clase: {len(class_to_idx)}")
    print(f"Număr fișiere/grupuri: {len(np.unique(groups))}")

    # ---------------------------------------------------------------
    # Feature extraction
    # ---------------------------------------------------------------
    X_features = extract_features(epochs_data)

    print(f"Formă caracteristici: {X_features.shape}")

    # ---------------------------------------------------------------
    # Group-based train/test split
    #
    # All epochs from one EDF file stay in the same split.
    # This prevents data leakage between train and test.
    # ---------------------------------------------------------------
    group_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_indices, test_indices = next(
        group_splitter.split(
            X_features,
            y_labels,
            groups=groups
        )
    )

    X_train = X_features[train_indices]
    X_test = X_features[test_indices]

    y_train = y_labels[train_indices]
    y_test = y_labels[test_indices]

    groups_train = groups[train_indices]
    groups_test = groups[test_indices]

    print(f"\nEpoci pentru antrenare: {len(X_train)}")
    print(f"Epoci pentru test: {len(X_test)}")
    print(
        f"Grupuri pentru antrenare: "
        f"{len(np.unique(groups_train))}"
    )
    print(
        f"Grupuri pentru test: "
        f"{len(np.unique(groups_test))}"
    )

    common_groups = set(groups_train) & set(groups_test)

    print(
        f"Grupuri comune între train și test: "
        f"{len(common_groups)}"
    )

    if len(common_groups) > 0:
        raise RuntimeError(
            "Data leakage detectat: există grupuri comune "
            "între train și test."
        )

    # ---------------------------------------------------------------
    # Standardization
    # Fit the scaler only on the training data
    # ---------------------------------------------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------------------------------------------------------------
    # Convert to PyTorch tensors
    # ---------------------------------------------------------------
    X_train_t = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_t = torch.tensor(
        y_train,
        dtype=torch.long
    )

    X_test_t = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_test_t = torch.tensor(
        y_test,
        dtype=torch.long
    )

    # ---------------------------------------------------------------
    # DataLoader
    #
    # shuffle=True shuffles epochs inside the training set.
    # The group-based split above ensures that test groups remain
    # completely separate.
    # ---------------------------------------------------------------
    train_dataset = TensorDataset(
        X_train_t,
        y_train_t
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    input_dim = X_features.shape[1]

    model = EEGMLP(
        input_size=input_dim,
        hidden_size=64,
        num_classes=len(class_to_idx)
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.005
    )
    
    model.train()
    history={
        "train_loss": [], "test_loss": [],
        "train_acc": [], "test_acc": [],
        "test_precision": [], "test_recall": [], "test_f1": [],
        "test_kappa": []
    }

    print("\nÎncepe antrenarea...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)

        # Evaluare la fiecare epocă
        train_metrics = evaluate(model, X_train_t, y_train_t)
        test_metrics = evaluate(model, X_test_t, y_test_t)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["test_acc"].append(test_metrics["accuracy"])
        history["test_precision"].append(test_metrics["precision"])
        history["test_recall"].append(test_metrics["recall"])
        history["test_f1"].append(test_metrics["f1"])
        history["test_kappa"].append(test_metrics["kappa"])

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoca {epoch + 1}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Acc: {test_metrics['accuracy']*100:.2f}% | "
                f"Test F1: {test_metrics['f1']:.4f}"
                )


    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------
    model.eval()

    with torch.no_grad():
        test_outputs = model(X_test_t)

        probabilities = torch.softmax(
            test_outputs,
            dim=1
        )

        predictions = torch.argmax(
            test_outputs,
            dim=1
        )

    # Convert tensors to NumPy arrays
    y_true = y_test_t.cpu().numpy()
    y_pred = predictions.cpu().numpy()

    # ---------------------------------------------------------------
    # Accuracy
    # ---------------------------------------------------------------
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nFULL DATASET")
    print(f"\nAcuratețe pe test: {accuracy * 100:.2f}%")

    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    # ---------------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------------
    target_names = [
        name
        for name, _ in sorted(
            class_to_idx.items(),
            key=lambda item: item[1]
        )
    ]

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)

    # ---------------------------------------------------------------
    # Cohen's kappa
    # ---------------------------------------------------------------
    kappa = cohen_kappa_score(
        y_true,
        y_pred
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cohen's κ: {kappa:.4f}")

    # ---------------------------------------------------------------
    # ROC-AUC for binary classification
    # ---------------------------------------------------------------
    auc = None

    if len(class_to_idx) == 2:
        positive_probabilities = (
            probabilities[:, 1]
            .cpu()
            .numpy()
        )

        auc = roc_auc_score(
            y_true,
            positive_probabilities
        )

        print(f"ROC-AUC: {auc:.4f}")

    # ---------------------------------------------------------------
    # Precision, recall and F1
    # ---------------------------------------------------------------
    prec_macro, rec_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    prec_weighted, rec_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    print(
        "\nMacro-averaged "
        f"Precision: {prec_macro:.4f} | "
        f"Recall: {rec_macro:.4f} | "
        f"F1: {f1_macro:.4f}"
    )

    print(
        "Weighted-averaged "
        f"Precision: {prec_weighted:.4f} | "
        f"Recall: {rec_weighted:.4f} | "
        f"F1: {f1_weighted:.4f}"
    )

    # ---------------------------------------------------------------
    # Save all metrics to a single CSV for easy inclusion in the
    # manuscript's Results tables
    # ---------------------------------------------------------------
    metrics_summary = {
        "accuracy": accuracy,
        "cohen_kappa": kappa,
        "roc_auc": auc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "weighted_precision": prec_weighted,
        "weighted_recall": rec_weighted,
        "weighted_f1": f1_weighted,
        "train_epochs": len(X_train),
        "test_epochs": len(X_test),
        "train_groups": len(np.unique(groups_train)),
        "test_groups": len(np.unique(groups_test)),
    }
    metrics_df = pd.DataFrame([metrics_summary])
    metrics_csv_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved metrics summary to: {metrics_csv_path}")

    # Per-class report also saved as CSV (precision/recall/F1/support per class)
    report_dict = classification_report(
        y_true, y_pred, target_names=target_names,
        digits=4, zero_division=0, output_dict=True,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = os.path.join(RESULTS_DIR, "classification_report.csv")
    report_df.to_csv(report_csv_path)
    print(f"Saved per-class classification report to: {report_csv_path}")

    # ---------------------------------------------------------------
    # SHAP interpretability analysis
    # ---------------------------------------------------------------
    feature_names = build_feature_names(CHANNELS)
     # Grafic metrici pentru toate epocile
    plot_metrics_history(history, RESULTS_DIR)

    # Grafic matrice de confuzie (final)
    plot_confusion_matrix_chart(y_true, y_pred, target_names, RESULTS_DIR)

    # Salvare istoric + model
    np.save(os.path.join(RESULTS_DIR, "history.npy"), history, allow_pickle=True)
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "model.pth"))

    run_shap_analysis(
        model=model,
        X_train_scaled=X_train,
        X_test_scaled=X_test,
        feature_names=feature_names,
        class_names=target_names,
        output_dir=SHAP_OUTPUT_DIR,
    )
