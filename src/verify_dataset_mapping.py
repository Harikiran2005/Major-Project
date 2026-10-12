
import os
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SPLIT_DIR = os.path.join(PROJECT_ROOT, "data", "splits")
COVID_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "covid_25d_v2")
CANCER_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "cancer_25d")

SPLITS = {
    "train": os.path.join(SPLIT_DIR, "train_final.csv"),
    "val": os.path.join(SPLIT_DIR, "val_final.csv"),
    "test": os.path.join(SPLIT_DIR, "test_final.csv"),
}

LABELS = {
    "normal": 0,
    "covid": 1,
    "cap": 2,
    "cancer": 3,
}

def load_split(path):
    df = pd.read_csv(path)
    required = {"patient_id", "class_name", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["class_name"] = df["class_name"].astype(str).str.strip().str.lower()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)

    for _, row in df.iterrows():
        expected = LABELS.get(row["class_name"])
        if expected is None or row["label"] != expected:
            raise ValueError(
                f"Invalid class/label mapping: "
                f"{row['patient_id']}, {row['class_name']}, {row['label']}"
            )

    if df["patient_id"].duplicated().any():
        print(f"WARNING: duplicate patient IDs in {os.path.basename(path)}")

    return df

def patient_windows(df, class_name):
    folder = os.path.join(COVID_DATA_DIR, class_name)
    if not os.path.isdir(folder):
        return []

    ids = set(df.loc[df["class_name"] == class_name, "patient_id"])
    matches = []

    for filename in os.listdir(folder):
        if filename.endswith(".npy"):
            for patient_id in ids:
                if filename.startswith(patient_id + "_"):
                    matches.append((patient_id, os.path.join(folder, filename)))
                    break

    return matches

def verify_array(path):
    arr = np.load(path)
    return (
        arr.shape == (3, 224, 224)
        and arr.dtype == np.float32
        and np.isfinite(arr).all()
        and float(arr.min()) >= 0
        and float(arr.max()) <= 1
    )

print("=" * 60)
print("V2 DATASET MAPPING VERIFICATION")
print("=" * 60)

splits = {name: load_split(path) for name, path in SPLITS.items()}

# Check patient-level separation
patient_sets = {
    name: set(df["patient_id"])
    for name, df in splits.items()
}

for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    overlap = patient_sets[a] & patient_sets[b]
    print(f"{a}/{b} patient overlap: {len(overlap)}")
    if overlap:
        print("Overlapping IDs:", sorted(overlap)[:10])
        raise ValueError("Patient leakage detected!")

# Check class counts and map every split to actual files
for split_name, df in splits.items():
    print(f"\n{split_name.upper()} patient counts:")
    print(df["class_name"].value_counts().to_string())

    total = 0
    for class_name in ["normal", "covid", "cap"]:
        matches = patient_windows(df, class_name)
        count = len(matches)
        total += count
        print(f"  {class_name}: {count} windows")

        # Check a few actual arrays for each class
        for _, path in matches[:5]:
            if not verify_array(path):
                raise ValueError(f"Invalid V2 array: {path}")

    cancer_df = df[df["class_name"] == "cancer"]
    cancer_found = 0

    for patient_id in cancer_df["patient_id"]:
        path = os.path.join(CANCER_DATA_DIR, f"{patient_id}_25d.npy")
        if os.path.isfile(path):
            cancer_found += 1
            if not verify_array(path):
                raise ValueError(f"Invalid cancer array: {path}")

    missing_cancer = len(cancer_df) - cancer_found
    total += cancer_found

    print(f"  cancer: {cancer_found} files; {missing_cancer} missing")
    print(f"  TOTAL mapped samples: {total}")

    if missing_cancer:
        raise ValueError(f"Missing cancer files in {split_name} split")

# Check V2 folder totals
print("\nV2 folder totals:")
for class_name in ["normal", "covid", "cap"]:
    folder = os.path.join(COVID_DATA_DIR, class_name)
    count = sum(
        1 for f in os.listdir(folder)
        if f.endswith(".npy")
    )
    print(f"  {class_name}: {count}")

print("\nPASS: no patient overlap, labels are consistent, and sampled arrays are valid.")
