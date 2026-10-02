"""
Fine-tune AraBERT for Arabic dialect identification.
Supports both 26-city and 5-group classification.
Run: python src/models/train.py
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from pathlib import Path
from torch.utils.data    import Dataset, DataLoader
from torch.optim         import AdamW
from transformers        import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics     import (
    accuracy_score,
    f1_score,
    classification_report
)

# ── PATHS ─────────────────────────────────────────────────────
Path("outputs/models").mkdir(parents=True, exist_ok=True)
Path("outputs/evaluation").mkdir(parents=True, exist_ok=True)


# ── CONFIG ────────────────────────────────────────────────────
CONFIG = {
    # Model
    "model_name":      "aubmindlab/bert-base-arabertv02",
    # AraBERT v2 — best for Arabic dialect tasks
    # Fallback: "asafaya/bert-mini-arabic" (much smaller, faster)

    # Task — choose one:
    "task":            "group",
    # "city"  → 26-class city-level classification
    # "group" → 5-class dialect group classification
    # START WITH "group" — faster, easier to debug
    # Switch to "city" after group works

    # Training
    "max_length":      64,
    # MADAR sentences average 6 words
    # 64 tokens is more than enough
    # Shorter = faster training

    "batch_size":      32,
    # Increase if your GPU has memory
    # Decrease to 16 if you get out-of-memory errors

    "num_epochs":      3,
    # 3 epochs is standard for BERT fine-tuning
    # Usually converges by epoch 2-3

    "learning_rate":   2e-5,
    # Standard for BERT fine-tuning
    # Range: 1e-5 to 5e-5

    "warmup_ratio":    0.1,
    # 10% of steps used for learning rate warmup
    # Prevents early training instability

    "weight_decay":    0.01,
    # L2 regularisation to prevent overfitting

    "use_class_weights": True,
    # Use computed class weights to handle imbalance

    # Paths
    "train_path":      "data/splits/train.csv",
    "val_path":        "data/splits/val.csv",
    "test_path":       "data/splits/test.csv",
    "weights_path":    "data/splits/class_weights.json",
    "mappings_path":   "data/splits/label_mappings.json",
    "output_dir":      "outputs/models",

    # MLflow
    "experiment_name": "arabic-dialect-identification",
    "run_name":        "arabert-dialect-group",
}


# ── DATASET CLASS ─────────────────────────────────────────────
class DialectDataset(Dataset):
    """
    PyTorch Dataset for Arabic dialect classification.
    Tokenises text on the fly during training.
    """

    def __init__(
        self,
        texts:      list,
        labels:     list,
        tokenizer,
        max_length: int = 64
    ):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text  = str(self.texts[idx])
        label = int(self.labels[idx])

        # Tokenise
        encoding = self.tokenizer(
            text,
            max_length      = self.max_length,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt"
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(label, dtype=torch.long)
        }


# ── TRAINING FUNCTIONS ────────────────────────────────────────
def train_one_epoch(
    model,
    dataloader,
    optimiser,
    scheduler,
    criterion,
    device
) -> tuple:
    """Train for one epoch. Returns avg loss and accuracy."""
    model.train()
    total_loss  = 0
    all_preds   = []
    all_labels  = []

    for batch_idx, batch in enumerate(dataloader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Forward pass
        optimiser.zero_grad()
        outputs = model(
            input_ids      = input_ids,
            attention_mask = attention_mask
        )
        logits = outputs.logits

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # Progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"    Batch {batch_idx+1}/{len(dataloader)}"
                  f"  Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def evaluate(
    model,
    dataloader,
    criterion,
    device
) -> tuple:
    """Evaluate model. Returns loss, accuracy, f1."""
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids      = input_ids,
                attention_mask = attention_mask
            )
            loss = criterion(outputs.logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds,
                        average="macro",
                        zero_division=0)
    return avg_loss, accuracy, f1, all_preds, all_labels


# ── MAIN TRAINING FUNCTION ────────────────────────────────────
def train():
    print("=" * 60)
    print("ARABIC DIALECT IDENTIFICATION — TRAINING")
    print("=" * 60)
    print(f"Task:      {CONFIG['task']} classification")
    print(f"Model:     {CONFIG['model_name']}")
    print(f"Epochs:    {CONFIG['num_epochs']}")
    print(f"Batch:     {CONFIG['batch_size']}")
    print(f"Max len:   {CONFIG['max_length']}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device:    {device}")
    if device.type == "cpu":
        print("WARNING: Training on CPU is slow.")
        print("         Recommend reducing batch for testing:")
        print("         Set batch_size=8, num_epochs=1 first.")

    # Load data
    print("\n[1/6] Loading data...")
    train_df = pd.read_csv(CONFIG["train_path"],
                           encoding="utf-8-sig")
    val_df   = pd.read_csv(CONFIG["val_path"],
                           encoding="utf-8-sig")
    test_df  = pd.read_csv(CONFIG["test_path"],
                           encoding="utf-8-sig")

    # Select label column based on task
    if CONFIG["task"] == "city":
        label_col   = "city_label"
        num_labels  = 26
        label_names = sorted(train_df["city_code"].unique())
    else:
        label_col   = "group_label"
        num_labels  = 5
        label_names = sorted(train_df["dialect_group"].unique())

    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")
    print(f"  Classes: {num_labels}")

    # Load class weights
    weights_tensor = None
    if CONFIG["use_class_weights"]:
        with open(CONFIG["weights_path"]) as f:
            weights_data = json.load(f)
        key = (f"{CONFIG['task']}_weights"
               if CONFIG["task"] == "city"
               else "group_weights")
        weights_dict   = weights_data[key]
        weights_list   = [weights_dict[str(i)]
                          for i in range(num_labels)]
        weights_tensor = torch.tensor(
            weights_list, dtype=torch.float
        ).to(device)
        print(f"  Class weights loaded: {num_labels} classes")

    # Load tokeniser
    print(f"\n[2/6] Loading tokeniser: {CONFIG['model_name']}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            CONFIG["model_name"]
        )
        print("  Tokeniser loaded")
    except Exception as e:
        print(f"  AraBERT failed: {e}")
        print("  Falling back to multilingual BERT...")
        CONFIG["model_name"] = "bert-base-multilingual-cased"
        tokenizer = AutoTokenizer.from_pretrained(
            CONFIG["model_name"]
        )
        print("  Multilingual BERT tokeniser loaded")

    # Create datasets
    print("\n[3/6] Creating datasets...")
    train_dataset = DialectDataset(
        texts      = train_df["text"].tolist(),
        labels     = train_df[label_col].tolist(),
        tokenizer  = tokenizer,
        max_length = CONFIG["max_length"]
    )
    val_dataset = DialectDataset(
        texts      = val_df["text"].tolist(),
        labels     = val_df[label_col].tolist(),
        tokenizer  = tokenizer,
        max_length = CONFIG["max_length"]
    )
    test_dataset = DialectDataset(
        texts      = test_df["text"].tolist(),
        labels     = test_df[label_col].tolist(),
        tokenizer  = tokenizer,
        max_length = CONFIG["max_length"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size = CONFIG["batch_size"],
        shuffle    = True,
        num_workers = 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = CONFIG["batch_size"],
        shuffle    = False,
        num_workers = 0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = CONFIG["batch_size"],
        shuffle    = False,
        num_workers = 0
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # Load model
    print(f"\n[4/6] Loading model: {CONFIG['model_name']}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels           = num_labels,
        ignore_mismatched_sizes = True
    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Optimiser and scheduler
    optimiser = AdamW(
        model.parameters(),
        lr           = CONFIG["learning_rate"],
        weight_decay = CONFIG["weight_decay"]
    )
    total_steps   = len(train_loader) * CONFIG["num_epochs"]
    warmup_steps  = int(total_steps * CONFIG["warmup_ratio"])
    scheduler     = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps
    )
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    print(f"  Total training steps: {total_steps:,}")
    print(f"  Warmup steps:         {warmup_steps:,}")

    # MLflow tracking
    print(f"\n[5/6] Training with MLflow tracking...")
    mlflow.set_experiment(CONFIG["experiment_name"])

    best_val_f1    = 0.0
    best_model_dir = Path(CONFIG["output_dir"]) / "best_model"

    train_losses = []
    val_losses   = []
    val_f1s      = []

    with mlflow.start_run(run_name=CONFIG["run_name"]) as run:
        # Log config
        mlflow.log_params({
            "model_name":    CONFIG["model_name"],
            "task":          CONFIG["task"],
            "num_labels":    num_labels,
            "batch_size":    CONFIG["batch_size"],
            "num_epochs":    CONFIG["num_epochs"],
            "learning_rate": CONFIG["learning_rate"],
            "max_length":    CONFIG["max_length"],
        })

        print(f"  MLflow run: {run.info.run_id[:8]}...")
        print()

        for epoch in range(CONFIG["num_epochs"]):
            print(f"  EPOCH {epoch+1}/{CONFIG['num_epochs']}")
            print(f"  {'─'*50}")

            # Train
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimiser,
                scheduler, criterion, device
            )
            train_time = time.time() - t0

            # Validate
            val_loss, val_acc, val_f1, _, _ = evaluate(
                model, val_loader, criterion, device
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_f1s.append(val_f1)

            # Log to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc":  train_acc,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
                "val_f1":     val_f1,
            }, step=epoch)

            print(f"  Train Loss: {train_loss:.4f}"
                  f"  Train Acc:  {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}"
                  f"  Val Acc:    {val_acc:.4f}"
                  f"  Val F1:     {val_f1:.4f}")
            print(f"  Time:       {train_time:.0f}s")

            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(best_model_dir))
                tokenizer.save_pretrained(str(best_model_dir))
                print(f"  ✅ Best model saved (F1={val_f1:.4f})")
            print()

        # Final evaluation on test set
        print("[6/6] Final evaluation on test set...")
        test_loss, test_acc, test_f1, preds, labels = evaluate(
            model, test_loader, criterion, device
        )

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_acc":  test_acc,
            "test_f1":   test_f1,
        })

        print()
        print("=" * 60)
        print("FINAL TEST RESULTS")
        print("=" * 60)
        print(f"Test Accuracy: {test_acc:.4f}  ({test_acc*100:.2f}%)")
        print(f"Test F1 Macro: {test_f1:.4f}  ({test_f1*100:.2f}%)")
        print(f"Best Val F1:   {best_val_f1:.4f}")

        # Classification report
        print("\nDetailed Classification Report:")
        print(classification_report(
            labels, preds,
            target_names = label_names,
            digits       = 4,
            zero_division = 0
        ))

        # Save results
        results = {
            "task":          CONFIG["task"],
            "model":         CONFIG["model_name"],
            "test_accuracy": float(test_acc),
            "test_f1_macro": float(test_f1),
            "best_val_f1":   float(best_val_f1),
            "num_classes":   num_labels,
            "train_samples": len(train_df),
            "val_samples":   len(val_df),
            "test_samples":  len(test_df),
        }
        results_path = "outputs/evaluation/results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved: {results_path}")

        mlflow.log_artifact(results_path)
        mlflow.log_artifact(str(best_model_dir))

    print(f"\nNext step: python src/models/evaluate.py")


# ── RUN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    train()