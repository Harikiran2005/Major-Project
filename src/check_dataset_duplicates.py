
import pandas as pd
from pathlib import Path

split_dir = Path("data/splits")

files = {
    "train": split_dir / "train_final.csv",
    "val": split_dir / "val_final.csv",
    "test": split_dir / "test_final.csv",
}

data = {}

for name, path in files.items():
    df = pd.read_csv(path)
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    data[name] = df

    print(f"\n{name.upper()}")
    print("CSV rows:", len(df))
    print("Unique patient IDs:", df["patient_id"].nunique())
    print("\nClass counts:")
    print(df["class_name"].value_counts().to_string())

print("\nPATIENT ID OVERLAP")
for first, second in [
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
]:
    a = set(data[first]["patient_id"])
    b = set(data[second]["patient_id"])
    overlap = sorted(a & b)

    print(f"{first} vs {second}: {len(overlap)} shared IDs")
    if overlap:
        print("Examples:", overlap[:10])

print("\nLABEL CONSISTENCY")
for name, df in data.items():
    mismatches = df[
        (
            (df["class_name"] == "normal") & (df["label"] != 0)
        ) | (
            (df["class_name"] == "covid") & (df["label"] != 1)
        ) | (
            (df["class_name"] == "cap") & (df["label"] != 2)
        ) | (
            (df["class_name"] == "cancer") & (df["label"] != 3)
        )
    ]

    print(f"{name}: {len(mismatches)} mismatched rows")
