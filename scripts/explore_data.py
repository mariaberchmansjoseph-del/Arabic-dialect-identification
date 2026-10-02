"""
Exploratory Data Analysis for MADAR Arabic Dialect Dataset.
Run: python scripts/explore_data.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

Path("outputs/evaluation").mkdir(parents=True, exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────
print("=" * 60)
print("MADAR ARABIC DIALECT DATASET — EDA")
print("=" * 60)

path = "data/raw/dialect_data.csv"
if not Path(path).exists():
    print(f"ERROR: {path} not found.")
    print("Run: python scripts/load_madar.py first")
    exit()

df = pd.read_csv(path, encoding="utf-8-sig")
print(f"\nLoaded: {path}")
print(f"Shape:  {df.shape}")


# ── SECTION 1: BASIC INFO ─────────────────────────────────────
print("\n" + "─" * 60)
print("SECTION 1: BASIC INFO")
print("─" * 60)
print(f"Total rows:     {len(df):,}")
print(f"Columns:        {list(df.columns)}")
print(f"Null values:\n{df.isnull().sum().to_string()}")
print(f"Duplicate rows: {df.duplicated().sum():,}")


# ── SECTION 2: CLASS DISTRIBUTION ────────────────────────────
print("\n" + "─" * 60)
print("SECTION 2: CLASS DISTRIBUTION")
print("─" * 60)

print("\nBy DIALECT GROUP (5 classes):")
group_counts = df["dialect_group"].value_counts()
for label, count in group_counts.items():
    pct = count / len(df) * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:<12} {count:>8,}  {pct:>5.1f}%  {bar}")

print("\nBy CITY (26 classes):")
city_counts = df["city_code"].value_counts()
print(f"{'City':<6} {'City Name':<15} {'Count':>8} {'Pct':>6}")
print("-" * 40)
for code, count in city_counts.items():
    name = df[df["city_code"] == code]["city_name"].iloc[0]
    pct  = count / len(df) * 100
    print(f"  {code:<6} {name:<15} {count:>8,} {pct:>5.1f}%")

max_count = city_counts.max()
min_count = city_counts.min()
ratio     = max_count / min_count
print(f"\nImbalance ratio (max/min): {ratio:.1f}x")
if ratio > 5:
    print("WARNING: High imbalance — use class weights")
elif ratio > 2:
    print("MODERATE imbalance — recommend class weights")
else:
    print("BALANCED — no special handling needed")


# ── SECTION 3: SPLIT DISTRIBUTION ────────────────────────────
print("\n" + "─" * 60)
print("SECTION 3: SPLIT DISTRIBUTION")
print("─" * 60)
print(df["split"].value_counts().to_string())

print("\nTrain cities (have training data):")
train_cities = df[df["split"] == "train"]["city_name"].unique()
print(f"  {sorted(train_cities)}")

print("\nTest-only cities (no training data in original split):")
all_cities   = set(df["city_name"].unique())
test_only    = all_cities - set(train_cities)
print(f"  {sorted(test_only)}")

print("\nNOTE: We will pool all data and create our own")
print("      train/val/test split to use all 26 cities.")


# ── SECTION 4: TEXT LENGTH ────────────────────────────────────
print("\n" + "─" * 60)
print("SECTION 4: TEXT LENGTH ANALYSIS")
print("─" * 60)

df["char_count"] = df["text"].astype(str).str.len()
df["word_count"] = df["text"].astype(str).str.split().str.len()
df["arabic_chars"] = df["text"].astype(str).apply(
    lambda t: sum(1 for c in t if "\u0600" <= c <= "\u06FF")
)
df["arabic_ratio"] = (
    df["arabic_chars"] / df["char_count"].clip(lower=1)
)

print(f"\nCharacter count:")
print(df["char_count"].describe().round(1).to_string())

print(f"\nWord count:")
print(df["word_count"].describe().round(1).to_string())

print(f"\nArabic character ratio:")
print(df["arabic_ratio"].describe().round(3).to_string())

low_arabic = (df["arabic_ratio"] < 0.5).sum()
print(f"\nRows with < 50% Arabic: {low_arabic:,}")


# ── SECTION 5: LENGTH PER DIALECT GROUP ──────────────────────
print("\n" + "─" * 60)
print("SECTION 5: WORD COUNT PER DIALECT GROUP")
print("─" * 60)
stats = df.groupby("dialect_group")["word_count"].agg(
    ["mean", "median", "min", "max"]
).round(1)
print(stats.to_string())


# ── SECTION 6: SAMPLE TEXTS ───────────────────────────────────
print("\n" + "─" * 60)
print("SECTION 6: SAMPLE TEXTS — SAME SENTENCE IN 5 GROUPS")
print("─" * 60)
print("(Sentence ID 0 translated into each dialect)\n")

sent_0 = df[df["text"].str.len() > 10].groupby(
    "dialect_group"
).first()

for group, row in sent_0.iterrows():
    print(f"  [{group}] ({row['city_name']})")
    print(f"  {row['text']}")
    print()


# ── SECTION 7: QUALITY CHECKS ─────────────────────────────────
print("\n" + "─" * 60)
print("SECTION 7: QUALITY CHECKS")
print("─" * 60)

has_url     = df["text"].str.contains(r"http|www", regex=True,
                                       case=False).sum()
has_mention = df["text"].str.contains("@").sum()
has_hashtag = df["text"].str.contains("#").sum()
short_texts = (df["word_count"] < 2).sum()
empty_texts = (df["text"].str.strip().str.len() == 0).sum()

print(f"Contains URLs:       {has_url:,}")
print(f"Contains @mentions:  {has_mention:,}")
print(f"Contains #hashtags:  {has_hashtag:,}")
print(f"Very short (< 2w):   {short_texts:,}")
print(f"Empty texts:         {empty_texts:,}")

total_issues = has_url + has_mention + short_texts + empty_texts
quality_pct  = (1 - total_issues / len(df)) * 100
print(f"\nQuality score: {quality_pct:.1f}%")
if quality_pct >= 95:
    print("EXCELLENT — minimal preprocessing needed")
elif quality_pct >= 85:
    print("GOOD — light preprocessing needed")
else:
    print("NEEDS CLEANING — significant preprocessing needed")


# ── SECTION 8: CHARTS ─────────────────────────────────────────
print("\n" + "─" * 60)
print("SECTION 8: GENERATING CHARTS")
print("─" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "MADAR Arabic Dialect Dataset — EDA",
    fontsize=14, fontweight="bold"
)

# Chart 1: Dialect group distribution
ax = axes[0, 0]
group_counts = df["dialect_group"].value_counts()
colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
bars = ax.bar(
    group_counts.index,
    group_counts.values,
    color     = colors[:len(group_counts)],
    edgecolor = "white",
    linewidth = 0.8
)
ax.set_title("Samples per Dialect Group", fontweight="bold")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=15)
for bar, val in zip(bars, group_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 200,
        f"{val:,}",
        ha="center", va="bottom", fontsize=9
    )

# Chart 2: City distribution
ax = axes[0, 1]
city_counts = df["city_code"].value_counts()
colors_city = plt.cm.tab20(np.linspace(0, 1, len(city_counts)))
ax.barh(
    range(len(city_counts)),
    city_counts.values,
    color     = colors_city,
    edgecolor = "white",
    linewidth = 0.5
)
ax.set_yticks(range(len(city_counts)))
ax.set_yticklabels(city_counts.index, fontsize=8)
ax.set_title("Samples per City", fontweight="bold")
ax.set_xlabel("Count")

# Chart 3: Word count distribution
ax = axes[1, 0]
ax.hist(
    df["word_count"].clip(upper=30),
    bins      = 25,
    color     = "#2196F3",
    edgecolor = "white",
    linewidth = 0.5
)
ax.axvline(
    df["word_count"].median(),
    color="red", linestyle="--",
    label=f"Median: {df['word_count'].median():.0f} words"
)
ax.set_title("Word Count Distribution", fontweight="bold")
ax.set_xlabel("Word count (capped at 30)")
ax.set_ylabel("Frequency")
ax.legend()

# Chart 4: Arabic ratio distribution
ax = axes[1, 1]
ax.hist(
    df["arabic_ratio"],
    bins      = 30,
    color     = "#4CAF50",
    edgecolor = "white",
    linewidth = 0.5
)
ax.set_title("Arabic Character Ratio", fontweight="bold")
ax.set_xlabel("Proportion of Arabic characters")
ax.set_ylabel("Frequency")
ax.axvline(
    df["arabic_ratio"].mean(),
    color="red", linestyle="--",
    label=f"Mean: {df['arabic_ratio'].mean():.2f}"
)
ax.legend()

plt.tight_layout()
chart_path = "outputs/evaluation/eda_charts.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Charts saved: {chart_path}")


# ── SECTION 9: WHAT HAPPENS NEXT ─────────────────────────────
print("\n" + "─" * 60)
print("SECTION 9: PREPROCESSING PLAN")
print("─" * 60)
print("""
MADAR is a professionally translated corpus.
Quality is very high. Preprocessing is minimal.

WHAT WE WILL DO:
  1. Diacritics removal (tashkeel)
     Some sentences have diacritics, most do not.
     Remove for consistency.

  2. Alef normalisation
     أ إ آ ٱ → ا
     Different translators may use different forms.

  3. Tatweel removal
     ـ (stretch character) → remove

  4. Whitespace normalisation
     Multiple spaces → single space
     Leading/trailing spaces → strip

  5. Teh Marbuta normalisation (optional)
     ة → ه
     Debate in Arabic NLP — we will test both.

WHAT WE WILL NOT DO:
  Remove punctuation (MADAR uses . and ، meaningfully)
  Remove short texts (all MADAR sentences are valid)
  Filter by Arabic ratio (already near 100%)

SPLIT STRATEGY:
  Pool all 111,193 sentences.
  Group by sentence ID (sent_id) before splitting.
  Same sentence ID goes to same split.
  This prevents data leakage between dialects.
  Split: 80% train / 10% val / 10% test.

Next step: python scripts/preprocess_data.py
""")

print("=" * 60)
print("EDA COMPLETE")
print("=" * 60)