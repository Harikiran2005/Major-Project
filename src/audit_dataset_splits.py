
from pathlib import Path
from collections import Counter
import pandas as pd

from dataset import LungDiseaseDataset

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits"
COVID_DIR = ROOT / "data" / "covid_25d_v2"
CANCER_DIR = ROOT / "data" / "cancer_25d"

split_names = ["train_final.csv", "val_final.csv", "test_final.csv"]
class_names = {0: "normal", 1: "covid", 2: "cap", 3: "cancer"}

patient_sets = {}

for filename in split_names:
    csv_path = SPLIT_DIR / filename
    df = pd.read_csv(csv_path)

    patient_sets[filename] = set(
        zip(df["class_name"].astype(str), df["patient_id"].astype(str))
    )

    dataset = LungDiseaseDataset(
        str(csv_path), str(COVID_DIR), str(CANCER_DIR)
    )

    counts = Counter(label for _, label in dataset.samples)

    print(f"\n--- {filename} ---")
    print("CSV patient rows:", len(df))
    print("Loaded windows/samples:", len(dataset))

    for label, name in class_names.items():
        print(f"{name}: {counts.get(label, 0)}")

overlaps = [
    ("train_final.csv", "val_final.csv"),
    ("train_final.csv", "test_final.csv"),
    ("val_final.csv", "test_final.csv"),
]

print("\n--- PATIENT OVERLAP CHECK ---")
for a, b in overlaps:
    shared = patient_sets[a] & patient_sets[b]
    print(f"{a} vs {b}: {len(shared)} shared class/patient IDs")
    if shared:
        print("Examples:", sorted(shared)[:10])
