"""
Load MADAR dataset from local TSV files.
Combines all 26 city dialect files into one clean CSV.
Run: python scripts/load_madar.py
"""

import pandas as pd
from pathlib import Path

# ── OUTPUT FOLDERS ────────────────────────────────────────────
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# ── CITY MAPPINGS ─────────────────────────────────────────────
CITY_TO_GROUP = {
    # Egyptian
    "CAI": "Egyptian",
    "ALX": "Egyptian",
    "ASW": "Egyptian",
    # Gulf
    "DOH": "Gulf",
    "RIY": "Gulf",
    "JED": "Gulf",
    "MUS": "Gulf",
    "SAN": "Gulf",
    "BAS": "Gulf",
    "BAG": "Gulf",
    "MOS": "Gulf",
    # Levantine
    "BEI": "Levantine",
    "DAM": "Levantine",
    "AMM": "Levantine",
    "JER": "Levantine",
    "SAL": "Levantine",
    "ALE": "Levantine",
    # Maghrebi
    "RAB": "Maghrebi",
    "FES": "Maghrebi",
    "TUN": "Maghrebi",
    "SFX": "Maghrebi",
    "ALG": "Maghrebi",
    "TRI": "Maghrebi",
    "BEN": "Maghrebi",
    # MSA and other
    "MSA": "MSA",
    "KHA": "MSA",
}

CITY_TO_ARABIC = {
    "CAI": "القاهرة",
    "ALX": "الإسكندرية",
    "ASW": "أسوان",
    "DOH": "الدوحة",
    "RIY": "الرياض",
    "JED": "جدة",
    "MUS": "مسقط",
    "SAN": "صنعاء",
    "BAS": "البصرة",
    "BAG": "بغداد",
    "MOS": "الموصل",
    "BEI": "بيروت",
    "DAM": "دمشق",
    "AMM": "عمان",
    "JER": "القدس",
    "SAL": "السلط",
    "ALE": "حلب",
    "RAB": "الرباط",
    "FES": "فاس",
    "TUN": "تونس",
    "SFX": "صفاقس",
    "ALG": "الجزائر",
    "TRI": "طرابلس",
    "BEN": "بنغازي",
    "MSA": "العربية الفصحى",
    "KHA": "الخرطوم",
}

# File name to city code mapping
FILENAME_TO_CODE = {
    "Cairo":      "CAI",
    "Alexandria": "ALX",
    "Aswan":      "ASW",
    "Doha":       "DOH",
    "Riyadh":     "RIY",
    "Jeddah":     "JED",
    "Muscat":     "MUS",
    "Sanaa":      "SAN",
    "Basra":      "BAS",
    "Baghdad":    "BAG",
    "Mosul":      "MOS",
    "Beirut":     "BEI",
    "Damascus":   "DAM",
    "Amman":      "AMM",
    "Jerusalem":  "JER",
    "Salt":       "SAL",
    "Aleppo":     "ALE",
    "Rabat":      "RAB",
    "Fes":        "FES",
    "Tunis":      "TUN",
    "Sfax":       "SFX",
    "Algiers":    "ALG",
    "Tripoli":    "TRI",
    "Benghazi":   "BEN",
    "MSA":        "MSA",
    "Khartoum":   "KHA",
}


def load_madar_files(madar_dir: str = "data/raw/madar") -> pd.DataFrame:
    """
    Load all MADAR TSV files and combine into one DataFrame.
    """
    madar_path = Path(madar_dir)

    if not madar_path.exists():
        print(f"ERROR: Folder not found: {madar_dir}")
        print("Please copy MADAR TSV files to data/raw/madar/")
        return None

    # Find all city TSV files
    tsv_files = [
        f for f in madar_path.glob("MADAR.corpus.*.tsv")
        if "English" not in f.name
        and "French" not in f.name
    ]

    if not tsv_files:
        print(f"ERROR: No TSV files found in {madar_dir}")
        print("Expected files like: MADAR.corpus.Cairo.tsv")
        return None

    print(f"Found {len(tsv_files)} city dialect files")
    print()

    all_rows = []

    for tsv_file in sorted(tsv_files):
        # Extract city name from filename
        # MADAR.corpus.Cairo.tsv → Cairo
        city_name = tsv_file.stem.replace("MADAR.corpus.", "")
        city_code = FILENAME_TO_CODE.get(city_name, city_name[:3].upper())

        try:
            df_city = pd.read_csv(
                tsv_file,
                sep       = "\t",
                encoding  = "utf-8",
                on_bad_lines = "skip"
            )

            # Column names vary slightly — find text column
            # MADAR uses: sentID.BTEC, split, lang, sent
            text_col  = None
            split_col = None
            lang_col  = None

            for col in df_city.columns:
                col_lower = col.lower()
                if col_lower in ["sent", "text", "sentence", "utterance"]:
                    text_col = col
                if col_lower == "split":
                    split_col = col
                if col_lower == "lang":
                    lang_col = col

            if text_col is None:
                # Last column is usually the text
                text_col = df_city.columns[-1]

            # Extract rows
            for _, row in df_city.iterrows():
                text  = str(row[text_col]).strip()
                split = str(row[split_col]).strip() if split_col else "train"
                lang  = str(row[lang_col]).strip()  if lang_col  else city_code

                # Use lang code from file if available (more accurate)
                code = FILENAME_TO_CODE.get(city_name, lang)

                if text and len(text) > 2 and text != "nan":
                    # Map split to simple train/test/dev
                    if "test" in split.lower():
                        simple_split = "test"
                    elif "dev" in split.lower():
                        simple_split = "dev"
                    else:
                        simple_split = "train"

                    all_rows.append({
                        "text":          text,
                        "city_code":     code,
                        "city_name":     city_name,
                        "city_arabic":   CITY_TO_ARABIC.get(code, city_name),
                        "dialect_group": CITY_TO_GROUP.get(code, "Other"),
                        "split":         simple_split,
                        "source":        "MADAR"
                    })

            rows_loaded = len(df_city)
            print(f"  ✅ {city_name:<15} ({city_code}) "
                  f"{rows_loaded:>6,} rows")

        except Exception as e:
            print(f"  ❌ {city_name:<15} ERROR: {str(e)[:60]}")

    if not all_rows:
        print("No data loaded. Check file format.")
        return None

    df = pd.DataFrame(all_rows)
    df = df[df["text"].str.len() > 2]
    df = df.drop_duplicates(subset=["text", "city_code"])
    df = df.reset_index(drop=True)

    return df


def print_summary(df: pd.DataFrame):
    """Print a clear summary of the loaded dataset."""
    print()
    print("=" * 60)
    print("MADAR DATASET SUMMARY")
    print("=" * 60)
    print(f"Total sentences:  {len(df):,}")
    print(f"Total cities:     {df['city_code'].nunique()}")
    print(f"Dialect groups:   {df['dialect_group'].nunique()}")
    print()

    print("SENTENCES PER CITY:")
    print(f"{'City':<15} {'Code':<6} {'Group':<12} {'Train':>7} "
          f"{'Test':>7} {'Total':>7}")
    print("-" * 60)

    city_stats = df.groupby(
        ["city_name", "city_code", "dialect_group", "split"]
    ).size().unstack(fill_value=0)

    for (city, code, group), row_data in df.groupby(
        ["city_name", "city_code", "dialect_group"]
    ):
        train = len(row_data[row_data["split"] == "train"])
        test  = len(row_data[row_data["split"] == "test"])
        total = len(row_data)
        print(f"{city:<15} {code:<6} {group:<12} "
              f"{train:>7,} {test:>7,} {total:>7,}")

    print("-" * 60)
    print()

    print("SENTENCES PER DIALECT GROUP:")
    print(f"{'Group':<12} {'Count':>8} {'Percent':>8}")
    print("-" * 32)
    group_counts = df["dialect_group"].value_counts()
    for group, count in group_counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"{group:<12} {count:>8,} {pct:>7.1f}% {bar}")

    print()
    print("SPLIT DISTRIBUTION:")
    print(df["split"].value_counts().to_string())

    print()
    print("SAMPLE SENTENCES PER DIALECT GROUP:")
    for group in df["dialect_group"].unique():
        sample = df[df["dialect_group"] == group]["text"].iloc[0]
        print(f"\n  [{group}]")
        print(f"  {sample[:100]}")


def save_dataset(df: pd.DataFrame):
    """Save the combined dataset."""
    # Full combined dataset
    full_path = "data/raw/dialect_data.csv"
    df.to_csv(full_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Full dataset saved:  {full_path}")

    # Separate train and test
    train_df = df[df["split"] == "train"]
    test_df  = df[df["split"] == "test"]
    dev_df   = df[df["split"] == "dev"]

    train_path = "data/processed/train_raw.csv"
    test_path  = "data/processed/test_raw.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path,   index=False, encoding="utf-8-sig")

    print(f"✅ Train split saved:   {train_path} ({len(train_df):,} rows)")
    print(f"✅ Test split saved:    {test_path}  ({len(test_df):,} rows)")

    if len(dev_df) > 0:
        dev_path = "data/processed/dev_raw.csv"
        dev_df.to_csv(dev_path, index=False, encoding="utf-8-sig")
        print(f"✅ Dev split saved:     {dev_path}   ({len(dev_df):,} rows)")


# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("LOADING MADAR ARABIC DIALECT DATASET")
    print("=" * 60)
    print()

    df = load_madar_files("data/raw/madar")

    if df is not None:
        print_summary(df)
        save_dataset(df)
        print()
        print("Next step: python scripts/explore_data.py")
    else:
        print("Loading failed. Check the steps above.")