"""
leak_mlpBBC.py -- BAD-COUNTER subjects only (EEGMAT), relaxed vs. one
mental-arithmetic stress state.

Data-handling condition: DATA-LEAK -- see leak_mlpBF.py for details.
Same pipeline as leak_mlpBF.py, pointed at the Bad-Counter subset.

NOTE ON NAMING: this script's own DATA_DIR/RESULTS_DIR use "BBC" for
Bad-Counter, while the non-leak/SHAP counterpart for the same subset is
named mshap_mlpBCC.py (data_BCC). Both refer to the same Bad-Counter
subject subset -- keep this in mind if you reorganise data_*/ folders.

Expects ./data_BBC/relaxed/*.edf and ./data_BBC/non-relaxed/*.edf.
Outputs: ./results_BBC/{model.pth, history.npy, confusion_matrix.png,
metrics_per_epoch.png}.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)

# ==========================================
# 1. CONFIGURARE ȘI PARAMETRI
# ==========================================
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

DATA_DIR = "data_BBC"
RESULTS_DIR = "results_BBC"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_EPOCHS = 1000


# ==========================================
# 2. FUNCȚIE PENTRU PROCESARE FIȘIER EDF
# ==========================================
def load_and_preprocess_edf(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=True)

    try:
        raw.pick(CHANNELS)
    except ValueError as e:
        print(f"Eroare canale în {file_path}: {e}")
        return None

    if raw.info["sfreq"] != SFREQ:
        raw.resample(SFREQ, verbose=False)

    raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)

    data, _ = raw[:, :]
    total_samples = data.shape[1]
    num_epochs = total_samples // EPOCH_LENGTH

    epochs = []
    for i in range(num_epochs):
        start = i * EPOCH_LENGTH
        end = start + EPOCH_LENGTH
        epochs.append(data[:, start:end])

    return np.array(epochs)


# ==========================================
# 3. ÎNCĂRCARE DATE CU ETICHETE DIN FOLDERE
# ==========================================
def load_dataset_from_folders(data_dir):
    X = []
    y = []
    class_names = sorted(
        [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    )

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    print("Clase detectate:", class_to_idx)

    for cls_name in class_names:
        class_dir = os.path.join(data_dir, cls_name)
        label = class_to_idx[cls_name]

        for filename in os.listdir(class_dir):
            if not filename.lower().endswith(".edf"):
                continue

            file_path = os.path.join(data_dir, cls_name, filename)
            epochs = load_and_preprocess_edf(file_path)

            if epochs is None or len(epochs) == 0:
                continue

            X.append(epochs)
            y.extend([label] * len(epochs))

    if len(X) == 0:
        return None, None, class_to_idx

    X = np.concatenate(X, axis=0)
    y = np.array(y)

    return X, y, class_to_idx


# ==========================================
# 4. EXTRAGERE CARACTERISTICI
# ==========================================
def extract_features(epochs_data):
    features = []
    for epoch in epochs_data:
        epoch_feats = []
        for channel_data in epoch:
            mean = np.mean(channel_data)
            std = np.std(channel_data)
            rms = np.sqrt(np.mean(channel_data ** 2))
            epoch_feats.extend([mean, std, rms])
        features.append(epoch_feats)
    return np.array(features)


# ==========================================
# 5. MODEL
# ==========================================
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
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 6. EVALUARE HELPER
# ==========================================
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


# ==========================================
# 7. GRAFICE
# ==========================================
def plot_metrics_history(history, results_dir):
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


def plot_confusion_matrix_chart(y_true, y_pred, target_names, results_dir):
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names,
                ax=axes[0])
    axes[0].set_title("Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Row-normalized
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # evităm împărțirea la zero
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


# ==========================================
# 8. ANTRENARE
# ==========================================
if __name__ == "__main__":
    epochs_data, y_labels, class_to_idx = load_dataset_from_folders(DATA_DIR)

    if epochs_data is None:
        print("Nu s-au găsit fișiere EDF.")
        raise SystemExit

    print(f"Epoci încărcate: {epochs_data.shape}")
    print(f"Număr clase: {len(class_to_idx)}")

    X_features = extract_features(epochs_data)
    print(f"Formă caracteristici: {X_features.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels,
        test_size=0.2, random_state=42, stratify=y_labels
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    input_dim = X_features.shape[1]
    num_classes = len(class_to_idx)
    model = EEGMLP(input_size=input_dim, hidden_size=64, num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    target_names = [name for name, _ in sorted(class_to_idx.items(),
                                               key=lambda x: x[1])]

    history = {
        "train_loss": [], "test_loss": [],
        "train_acc": [], "test_acc": [],
        "test_precision": [], "test_recall": [], "test_f1": [],
        "test_kappa": [],
    }

    print("\nÎncepe antrenarea...")
    for epoch in range(N_EPOCHS):
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

        # ---- evaluare la fiecare epocă ----
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

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoca {epoch+1}/{N_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_metrics['loss']:.4f} | "
                  f"Test Acc: {test_metrics['accuracy']*100:.2f}% | "
                  f"Test F1: {test_metrics['f1']:.4f}")

    # ==========================================
    # 9. REZULTATE FINALE + GRAFICE
    # ==========================================
    final_metrics = evaluate(model, X_test_t, y_test_t)
    y_true, y_pred = final_metrics["y_true"], final_metrics["y_pred"]

    print(f"\nAcuratețe finală pe test: {final_metrics['accuracy']*100:.2f}%")
    print(f"Cohen's κ: {final_metrics['kappa']:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # Grafic metrici pentru toate epocile
    plot_metrics_history(history, RESULTS_DIR)

    # Grafic matrice de confuzie (final)
    plot_confusion_matrix_chart(y_true, y_pred, target_names, RESULTS_DIR)

    # Salvare model + istoric
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "model.pth"))
    np.save(os.path.join(RESULTS_DIR, "history.npy"), history, allow_pickle=True)
    print("\nModel și istoric salvate în folderul 'results/'.")
