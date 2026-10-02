import pandas as pd
from pathlib import Path
from datasets import load_dataset

Path("data/raw").mkdir(parents=True, exist_ok=True)

CITY_TO_GROUP = {
    "CAI": "Egyptian",  "ALX": "Egyptian",  "ASW": "Egyptian",
    "DOH": "Gulf",      "ABU": "Gulf",      "DUB": "Gulf",
    "MUS": "Gulf",      "KWI": "Gulf",      "BAH": "Gulf",
    "RIY": "Gulf",      "JED": "Gulf",      "SAN": "Gulf",
    "ADE": "Gulf",
    "BEI": "Levantine", "DAM": "Levantine", "AMM": "Levantine",
    "JER": "Levantine", "SAL": "Levantine",
    "RAB": "Maghrebi",  "FES": "Maghrebi",  "TUN": "Maghrebi",
    "SFX": "Maghrebi",  "ALG": "Maghrebi",  "TRI": "Maghrebi",
    "MSA": "MSA",
}

CITY_TO_ARABIC = {
    "CAI": "القاهرة",    "ALX": "الإسكندرية", "ASW": "أسوان",
    "DOH": "الدوحة",     "ABU": "أبوظبي",     "DUB": "دبي",
    "MUS": "مسقط",       "KWI": "الكويت",     "BAH": "البحرين",
    "RIY": "الرياض",     "JED": "جدة",        "SAN": "صنعاء",
    "ADE": "عدن",        "BEI": "بيروت",      "DAM": "دمشق",
    "AMM": "عمان",       "JER": "القدس",      "SAL": "السلط",
    "RAB": "الرباط",     "FES": "فاس",        "TUN": "تونس",
    "SFX": "صفاقس",      "ALG": "الجزائر",    "TRI": "طرابلس",
    "MSA": "العربية الفصحى",
}


def try_load(name, config=None):
    try:
        if config:
            ds = load_dataset(name, config, trust_remote_code=True)
        else:
            ds = load_dataset(name, trust_remote_code=True)
        first = list(ds.keys())[0]
        print(f"  OK: {name} {config or ''}")
        print(f"      Splits:  {list(ds.keys())}")
        print(f"      Columns: {ds[first].column_names}")
        return ds
    except Exception as e:
        print(f"  FAIL: {name} {config or ''}")
        print(f"        {str(e)[:100]}")
        return None


def create_sample():
    print("Creating sample dataset...")
    samples = [
        ("عايز اروح البيت دلوقتي", "CAI"),
        ("ايه اللي بيحصل ده", "CAI"),
        ("انا مبسوط جدا النهارده", "CAI"),
        ("هو فين ده بقى", "CAI"),
        ("ازيك يا صاحبي", "ALX"),
        ("زمان ما شفتكش", "ALX"),
        ("الجو حلو النهارده", "ALX"),
        ("ابي اروح البيت الحين", "DUB"),
        ("انا مرتاح وايد اليوم", "DUB"),
        ("شلونك يا صديقي", "DOH"),
        ("عندي شغل وايد", "ABU"),
        ("ما فهمت شو تقصد", "KWI"),
        ("الجو حار اليوم وايد", "RIY"),
        ("كيف الاهل", "JED"),
        ("يا وليد وين رحت", "BAH"),
        ("بدي روح عالبيت هلق", "BEI"),
        ("شو في جديد", "DAM"),
        ("انا فرحان كتير هلق", "BEI"),
        ("وين رايح", "AMM"),
        ("كيفك يا صاحبي", "DAM"),
        ("ما فهمت شو قلت", "JER"),
        ("شو بتشتغل", "AMM"),
        ("بغيت نمشي للدار دابا", "RAB"),
        ("اش كاين الجديد", "FES"),
        ("انا فرحان بزاف هاد النهار", "TUN"),
        ("فين غادي", "RAB"),
        ("لاباس عليك", "ALG"),
        ("واش راك بخير", "RAB"),
        ("عندي خدمة بزاف", "TUN"),
        ("علاش ما جيتيش", "TRI"),
        ("اريد ان اذهب الى المنزل الان", "MSA"),
        ("ما الاخبار الجديدة", "MSA"),
        ("انا سعيد جدا اليوم", "MSA"),
        ("الى اين تذهب", "MSA"),
        ("كيف حالك يا صديقي", "MSA"),
        ("لدي عمل كثير", "MSA"),
        ("لم افهم ما قلته", "MSA"),
        ("الطقس جميل اليوم", "MSA"),
    ]
    rows = []
    for text, city in samples:
        rows.append({
            "text":          text,
            "city_code":     city,
            "city_arabic":   CITY_TO_ARABIC.get(city, city),
            "dialect_group": CITY_TO_GROUP.get(city, "Unknown"),
            "source":        "sample"
        })
    df = pd.DataFrame(rows)
    path = "data/raw/dialect_data.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path}")
    print(f"Rows:  {len(df)}")
    print(df["dialect_group"].value_counts().to_string())
    return df


print("=" * 55)
print("DOWNLOADING ARABIC DIALECT DATASET")
print("=" * 55)

dataset = None
attempts = [
    ("camel-lab/madar", "MADAR-26"),
    ("camel-lab/madar", "MADAR-6"),
    ("camel-lab/madar", None),
    ("arbml/madar",     None),
    ("arbml/nadi2023",  None),
]

for name, config in attempts:
    print(f"\nTrying: {name} {config or ''}...")
    dataset = try_load(name, config)
    if dataset:
        break

if dataset is None:
    print("\nAll downloads failed. Using sample data.")
    df = create_sample()
else:
    all_rows = []
    for split_name in dataset.keys():
        split = dataset[split_name]
        print(f"\nProcessing: {split_name} ({len(split)} rows)")
        for row in split:
            text = (
                row.get("text") or
                row.get("sentence") or
                row.get("utterance") or
                row.get("tweet") or ""
            )
            label = (
                row.get("label") or
                row.get("dialect") or
                row.get("city") or
                row.get("country") or ""
            )
            text  = str(text).strip()
            label = str(label).strip().upper()
            if text and label and len(text) > 2:
                all_rows.append({
                    "text":          text,
                    "city_code":     label,
                    "city_arabic":   CITY_TO_ARABIC.get(label, label),
                    "dialect_group": CITY_TO_GROUP.get(label, "Other"),
                    "split":         split_name,
                    "source":        "MADAR"
                })

    df = pd.DataFrame(all_rows)
    df = df[df["text"].str.len() > 2]
    df = df.drop_duplicates(subset=["text"])
    df = df.reset_index(drop=True)

    path = "data/raw/dialect_data.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*55}")
    print(f"COMPLETE")
    print(f"Total rows: {len(df):,}")
    print(f"\nCity distribution:")
    print(df["city_code"].value_counts().to_string())
    print(f"\nDialect group distribution:")
    print(df["dialect_group"].value_counts().to_string())
    print(f"\nSaved: data/raw/dialect_data.csv")

print(f"\nNext: python scripts/explore_data.py")

