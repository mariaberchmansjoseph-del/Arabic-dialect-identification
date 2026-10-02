"""
Generate confusion matrix and evaluation charts.
Run: python src/models/evaluate.py
"""

import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    classification_report
)

Path("outputs/evaluation").mkdir(parents=True, exist_ok=True)

# ── CONFIG ────────────────────────────────────────────────────
MODEL_DIR  = "outputs/models/best_model"
TEST_PATH  = "data/splits/test.csv"
MAP_PATH   = "data/splits/label_mappings.json"
TASK       = "group"
MAX_LENGTH = 64
BATCH_SIZE = 32

DIALECT_COLORS = {
    "Egyptian":  "#2196F3",
    "Gulf":      "#FF9800",
    "Levantine": "#4CAF50",
    "MSA":       "#9C27B0",
    "Maghrebi":  "#F44336",
}


# ── DATASET ───────────────────────────────────────────────────
class DialectDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length     = self.max_length,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(int(self.labels[idx]),
                                   dtype=torch.long)
        }


# ── LOAD MODEL ────────────────────────────────────────────────
def load_model(model_dir):
    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_dir
    )
    model.eval()
    return tokenizer, model


# ── RUN PREDICTIONS ───────────────────────────────────────────
def get_predictions(model, dataloader, device):
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 20 == 0:
                print(f"  Batch {i}/{len(dataloader)}")
            ids   = batch["input_ids"].to(device)
            masks = batch["attention_mask"].to(device)
            labs  = batch["labels"]

            outputs = model(input_ids=ids, attention_mask=masks)
            preds   = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.numpy())

    return np.array(all_preds), np.array(all_labels)


# ── CONFUSION MATRIX CHART ────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, label_names):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Arabic Dialect Identification — Confusion Matrix",
        fontsize=14, fontweight="bold"
    )

    # Chart 1: Raw counts
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=30, ha="right",
                       fontsize=10)
    ax.set_yticklabels(label_names, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Raw Counts", fontweight="bold")
    plt.colorbar(im, ax=ax)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:,}",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i,j] > thresh else "black",
                    fontweight="bold")

    # Chart 2: Percentage (per row = per true class)
    ax2 = axes[1]
    im2 = ax2.imshow(cm_pct, cmap="RdYlGn", vmin=0, vmax=100)
    ax2.set_xticks(range(len(label_names)))
    ax2.set_yticks(range(len(label_names)))
    ax2.set_xticklabels(label_names, rotation=30, ha="right",
                        fontsize=10)
    ax2.set_yticklabels(label_names, fontsize=10)
    ax2.set_xlabel("Predicted", fontsize=11)
    ax2.set_ylabel("True", fontsize=11)
    ax2.set_title("Row Percentages (%)", fontweight="bold")
    plt.colorbar(im2, ax=ax2)

    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            ax2.text(j, i, f"{cm_pct[i,j]:.1f}%",
                     ha="center", va="center", fontsize=9,
                     color="black", fontweight="bold")

    plt.tight_layout()
    path = "outputs/evaluation/confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Confusion matrix saved: {path}")


# ── F1 BAR CHART ──────────────────────────────────────────────
def plot_f1_bars(y_true, y_pred, label_names):
    f1s    = f1_score(y_true, y_pred, average=None,
                      zero_division=0)
    colors = [DIALECT_COLORS.get(n, "#607D8B")
              for n in label_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(label_names, f1s * 100,
                  color=colors, edgecolor="white",
                  linewidth=0.8)

    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val*100:.1f}%",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold"
        )

    # Overall F1 line
    macro_f1 = f1_score(y_true, y_pred,
                        average="macro", zero_division=0)
    ax.axhline(macro_f1 * 100, color="red",
               linestyle="--", linewidth=1.5,
               label=f"Macro F1: {macro_f1*100:.1f}%")

    ax.set_ylim(0, 105)
    ax.set_ylabel("F1 Score (%)", fontsize=12)
    ax.set_title(
        "Arabic Dialect Identification — F1 Score per Class",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")

    plt.tight_layout()
    path = "outputs/evaluation/f1_per_class.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ F1 chart saved: {path}")


# ── PRINT SUMMARY ─────────────────────────────────────────────
def print_summary(y_true, y_pred, label_names):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro",
                   zero_division=0)

    print()
    print("=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    print(f"Test Accuracy: {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Test F1 Macro: {f1:.4f}  ({f1*100:.2f}%)")
    print()
    print(classification_report(
        y_true, y_pred,
        target_names  = label_names,
        digits        = 4,
        zero_division = 0
    ))

    # Interesting confusions
    cm = confusion_matrix(y_true, y_pred)
    print("MOST COMMON CONFUSIONS:")
    print(f"{'True':<12} {'Predicted':<12} {'Count':>8}")
    print("-" * 35)
    confusions = []
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i,j], label_names[i],
                                   label_names[j]))
    confusions.sort(reverse=True)
    for count, true_l, pred_l in confusions[:8]:
        print(f"{true_l:<12} → {pred_l:<12} {count:>8,}")


# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("GENERATING EVALUATION CHARTS")
    print("=" * 55)

    # Load label mappings
    with open(MAP_PATH) as f:
        mappings = json.load(f)

    if TASK == "group":
        label_map  = mappings["label_to_group"]
        label_col  = "group_label"
    else:
        label_map  = mappings["label_to_city"]
        label_col  = "city_label"

    label_names = [label_map[str(i)]
                   for i in range(len(label_map))]
    print(f"Classes: {label_names}")

    # Load test data
    test_df = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    print(f"Test samples: {len(test_df):,}")

    # Load model
    device    = torch.device("cpu")
    tokenizer, model = load_model(MODEL_DIR)
    model.to(device)

    # Create dataset
    dataset = DialectDataset(
        texts      = test_df["text"].tolist(),
        labels     = test_df[label_col].tolist(),
        tokenizer  = tokenizer,
        max_length = MAX_LENGTH
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)

    # Get predictions
    print("\nRunning predictions on test set...")
    preds, labels = get_predictions(model, loader, device)

    # Print summary
    print_summary(labels, preds, label_names)

    # Generate charts
    print("\nGenerating charts...")
    plot_confusion_matrix(labels, preds, label_names)
    plot_f1_bars(labels, preds, label_names)

    print("\n" + "=" * 55)
    print("CHARTS SAVED TO: outputs/evaluation/")
    print("  confusion_matrix.png")
    print("  f1_per_class.png")
    print("=" * 55)