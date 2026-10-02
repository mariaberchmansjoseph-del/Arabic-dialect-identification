"""
Preprocess MADAR dataset for dialect identification.
Steps: normalise Arabic text → create proper split → save.
Run: python scripts/preprocess_data.py
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder

Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("data/splits").mkdir(parents=True, exist_ok=True)


# ── ARABIC TEXT NORMALISER ────────────────────────────────────
class ArabicNormaliser:
    """
    Minimal normaliser for MADAR dataset.
    MADAR is already clean so we do only what is necessary.
    """

    def __init__(self):
        # Diacritics (tashkeel) pattern
        self.diacritics = re.compile(
            r"[\u0617-\u061A\u064B-\u0652]"
        )
        # Alef variants → bare alef
        self.alef_map = {
            "\u0623": "\u0627",  # أ → ا
            "\u0625": "\u0627",  # إ → ا
            "\u0622": "\u0627",  # آ → ا
            "\u0671": "\u0627",  # ٱ → ا
        }
        # Tatweel (stretch character)
        self.tatweel = "\u0640"

    def normalise(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        # 1. Remove diacritics
        text = self.diacritics.sub("", text)

        # 2. Normalise alef variants
        for variant, base in self.alef_map.items():
            text = text.replace(variant, base)

        # 3. Remove tatweel
        text = text.replace(self.tatweel, "")

        # 4. Normalise whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def normalise_batch(self, texts: list) -> list:
        return [self.normalise(t) for t in texts]


def create_balanced_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a proper train/val/test split using sentence IDs.

    WHY WE DO THIS:
    MADAR is a parallel corpus — the same sentence appears
    in all 26 dialects. If sentence 42 goes to train for
    Cairo, it should also go to train for Doha, Beirut etc.

    If we split randomly by row, sentence 42 might be in
    train for Cairo but test for Doha. The model would then
    see the same meaning in training and be tested on it —
    this is DATA LEAKAGE and gives falsely high accuracy.

    CORRECT APPROACH:
    1. Get all unique sentence IDs.
    2. Split the sentence IDs (not the rows).
    3. Assign all rows with the same sentence ID
       to the same split.
    """

    print("Creating split by sentence ID (prevents data leakage)...")

    # MADAR uses sentID.BTEC in original files
    # We do not have that column after loading
    # So we group by text of MSA version as proxy
    # Actually: we use row index divided by 26 as sentence group
    # (since each group of 26 rows = same sentence in 26 dialects)

    # Simpler and correct: get unique sentences from one city
    # and use those IDs for splitting
    # Each sentence appears exactly once per city
    # Total rows = 26 cities × ~4,277 sentences = 111,193

    # Get all unique texts from Cairo (represents sentence IDs)
    # Since MADAR is parallel, sentence positions align
    cairo_texts = df[df["city_code"] == "CAI"]["text"].tolist()
    n_sentences = len(cairo_texts)

    print(f"  Unique sentence positions: ~{n_sentences:,}")

    # Create sentence position index
    # Group rows by their position within each city file
    df = df.copy()

    # Assign sentence group by sorting within each city
    # and numbering sequentially
    df["sent_group"] = df.groupby("city_code").cumcount()

    # Get unique sentence group IDs
    sent_groups = df["sent_group"].unique()

    # Split sentence groups 80/10/10
    n       = len(sent_groups)
    n_val   = int(n * 0.10)
    n_test  = int(n * 0.10)
    n_train = n - n_val - n_test

    np.random.seed(42)
    shuffled = np.random.permutation(sent_groups)

    train_ids = set(shuffled[:n_train])
    val_ids   = set(shuffled[n_train:n_train + n_val])
    test_ids  = set(shuffled[n_train + n_val:])

    # Assign splits
    def assign_split(sent_group):
        if sent_group in train_ids:
            return "train"
        elif sent_group in val_ids:
            return "val"
        else:
            return "test"

    df["new_split"] = df["sent_group"].apply(assign_split)

    print(f"  Train sentence groups: {len(train_ids):,}")
    print(f"  Val sentence groups:   {len(val_ids):,}")
    print(f"  Test sentence groups:  {len(test_ids):,}")

    return df


def encode_labels(df: pd.DataFrame):
    """
    Create numerical label encodings for both
    city-level (26 classes) and group-level (5 classes).
    """
    # City-level encoder (26 classes)
    city_encoder = LabelEncoder()
    df["city_label"] = city_encoder.fit_transform(df["city_code"])

    # Group-level encoder (5 classes)
    group_encoder = LabelEncoder()
    df["group_label"] = group_encoder.fit_transform(
        df["dialect_group"]
    )

    return df, city_encoder, group_encoder


def compute_class_weights(df: pd.DataFrame,
                           label_col: str) -> dict:
    """
    Compute class weights to handle imbalance.
    Higher weight = model penalised more for errors on that class.
    """
    from sklearn.utils.class_weight import compute_class_weight

    labels  = df[label_col].values
    classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight = "balanced",
        classes      = classes,
        y            = labels
    )

    return dict(zip(classes.tolist(), weights.tolist()))


def save_splits(df: pd.DataFrame,
                city_encoder,
                group_encoder):
    """Save train/val/test splits and label mappings."""
    import json

    train = df[df["new_split"] == "train"]
    val   = df[df["new_split"] == "val"]
    test  = df[df["new_split"] == "test"]

    # Save splits
    for name, split_df in [("train", train),
                            ("val",   val),
                            ("test",  test)]:
        path = f"data/splits/{name}.csv"
        split_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  ✅ {name:5}: {len(split_df):>7,} rows → {path}")

    # Save label mappings
    city_map = {
        int(i): label
        for i, label in enumerate(city_encoder.classes_)
    }
    group_map = {
        int(i): label
        for i, label in enumerate(group_encoder.classes_)
    }

    mappings = {
        "city_to_label":   {v: k for k, v in city_map.items()},
        "label_to_city":   city_map,
        "group_to_label":  {v: k for k, v in group_map.items()},
        "label_to_group":  group_map,
        "num_city_classes":  len(city_map),
        "num_group_classes": len(group_map)
    }

    map_path = "data/splits/label_mappings.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)
    print(f"  ✅ Label mappings → {map_path}")

    return train, val, test


def print_split_summary(train, val, test):
    """Print clear summary of the final splits."""
    print()
    print("=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    print(f"Train: {len(train):,} rows")
    print(f"Val:   {len(val):,} rows")
    print(f"Test:  {len(test):,} rows")
    print(f"Total: {len(train)+len(val)+len(test):,} rows")

    print("\nClass distribution in TRAIN (city level):")
    city_train = train["city_code"].value_counts()
    for code, count in city_train.items():
        pct = count / len(train) * 100
        print(f"  {code:<6} {count:>6,}  ({pct:.1f}%)")

    print("\nClass distribution in TRAIN (dialect group):")
    group_train = train["dialect_group"].value_counts()
    for group, count in group_train.items():
        pct = count / len(train) * 100
        bar = "█" * int(pct / 3)
        print(f"  {group:<12} {count:>7,}  ({pct:.1f}%)  {bar}")

    # Verify no leakage
    print("\nData leakage check:")
    train_groups = set(train["sent_group"].unique())
    val_groups   = set(val["sent_group"].unique())
    test_groups  = set(test["sent_group"].unique())

    train_val_overlap  = train_groups & val_groups
    train_test_overlap = train_groups & test_groups
    val_test_overlap   = val_groups   & test_groups

    print(f"  Train ∩ Val:  {len(train_val_overlap)} "
          f"{'✅ No leakage' if len(train_val_overlap)==0 else '❌ LEAKAGE!'}")
    print(f"  Train ∩ Test: {len(train_test_overlap)} "
          f"{'✅ No leakage' if len(train_test_overlap)==0 else '❌ LEAKAGE!'}")
    print(f"  Val ∩ Test:   {len(val_test_overlap)} "
          f"{'✅ No leakage' if len(val_test_overlap)==0 else '❌ LEAKAGE!'}")


# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("PREPROCESSING MADAR ARABIC DIALECT DATASET")
    print("=" * 60)

    # Load
    print("\n[1/5] Loading data...")
    df = pd.read_csv(
        "data/raw/dialect_data.csv",
        encoding="utf-8-sig"
    )
    print(f"  Loaded {len(df):,} rows")

    # Normalise text
    print("\n[2/5] Normalising Arabic text...")
    normaliser = ArabicNormaliser()
    original   = df["text"].iloc[0]
    df["text"] = normaliser.normalise_batch(df["text"].tolist())
    normalised = df["text"].iloc[0]
    print(f"  Before: {original}")
    print(f"  After:  {normalised}")
    print(f"  Processed {len(df):,} sentences")

    # Remove empty after normalisation
    before = len(df)
    df     = df[df["text"].str.len() > 0]
    df     = df.reset_index(drop=True)
    removed = before - len(df)
    print(f"  Removed {removed} empty sentences after normalisation")

    # Create split
    print("\n[3/5] Creating train/val/test split...")
    df = create_balanced_split(df)

    # Encode labels
    print("\n[4/5] Encoding labels...")
    df, city_encoder, group_encoder = encode_labels(df)
    print(f"  City classes:  {len(city_encoder.classes_)}")
    print(f"  Group classes: {len(group_encoder.classes_)}")
    print(f"  City labels:   {list(city_encoder.classes_)}")
    print(f"  Group labels:  {list(group_encoder.classes_)}")

    # Save
    print("\n[5/5] Saving splits...")
    train, val, test = save_splits(df, city_encoder, group_encoder)

    # Summary
    print_split_summary(train, val, test)

    # Class weights for training
    print("\nClass weights for training (city level):")
    city_weights = compute_class_weights(train, "city_label")
    for label, weight in sorted(city_weights.items()):
        city_name = city_encoder.classes_[label]
        print(f"  {city_name:<15} weight: {weight:.3f}")

    print("\nClass weights for training (group level):")
    group_weights = compute_class_weights(train, "group_label")
    for label, weight in sorted(group_weights.items()):
        group_name = group_encoder.classes_[label]
        print(f"  {group_name:<12} weight: {weight:.3f}")

    import json
    weights_path = "data/splits/class_weights.json"
    with open(weights_path, "w") as f:
        json.dump({
            "city_weights":  city_weights,
            "group_weights": group_weights
        }, f, indent=2)
    print(f"\n✅ Class weights saved: {weights_path}")
    print("\nNext step: python src/models/train.py")
