
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[1]

# Existing cancer patients in all three splits
existing_ids = set()

for split in ["train_final.csv", "val_final.csv", "test_final.csv"]:
    df = pd.read_csv(ROOT / "data" / "splits" / split)
    cancer = df[df["class_name"].astype(str).str.strip().str.lower() == "cancer"]
    existing_ids.update(cancer["patient_id"].astype(str).str.strip())

# QIN patient folder names
qin_root = ROOT / "data" / "qin_cancer_25d"
qin_ids = {
    p.name for p in qin_root.iterdir() if p.is_dir()
}

def canonical_id(value):
    """Normalize R_144, R0144 and R-0144 to R144."""
    value = value.strip().upper()
    match = re.fullmatch(r"R[_-]?0*(\d+)", value)
    if match:
        return "R" + str(int(match.group(1)))
    return value

existing_normalized = {}
for patient_id in existing_ids:
    existing_normalized.setdefault(
        canonical_id(patient_id), []
    ).append(patient_id)

matches = []
for qin_id in sorted(qin_ids):
    key = canonical_id(qin_id)
    if key in existing_normalized:
        matches.append((qin_id, existing_normalized[key]))

print("Existing cancer patients:", len(existing_ids))
print("QIN patients:", len(qin_ids))
print("Potential normalized ID matches:", len(matches))

for qin_id, existing in matches:
    print(f"QIN {qin_id} <-> existing {', '.join(existing)}")
