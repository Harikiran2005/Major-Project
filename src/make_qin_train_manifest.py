
from pathlib import Path
import pandas as pd

ROOT = Path("data/qin_cancer_25d")
OUT = Path("data/splits/qin_train_manifest.csv")

rows = []

for patient_dir in sorted(ROOT.iterdir()):
    if not patient_dir.is_dir():
        continue

    for file in sorted(patient_dir.glob("*.npy")):
        rows.append({
            "patient_id": patient_dir.name,
            "class_name": "cancer",
            "label": 3,
            "filepath": str(file.resolve()),
            "split": "train_only",
        })

df = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print("Manifest saved to:", OUT.resolve())
print("QIN patients:", df["patient_id"].nunique())
print("QIN windows:", len(df))
print("Classes:", df["class_name"].value_counts().to_dict())
print("Splits:", df["split"].value_counts().to_dict())
