
import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASETS = [
    ("normal", "meta_data_normal.csv", "Slice"),
    ("covid", "meta_data_covid.csv", "Slice"),
    ("cap", "meta_data_cap.csv", "Slices_x"),
]

for name, filename, slice_col in DATASETS:
    path = os.path.join(PROJECT_ROOT, "..", filename)

    print(f"\n--- {name.upper()} ---")

    if not os.path.exists(path):
        print("Missing metadata:", path)
        continue

    df = pd.read_csv(path, encoding="latin1")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    if not {"Patient ID", "File name", slice_col}.issubset(df.columns):
        print("Required columns missing; skipping checks.")
        continue

    df["Patient ID"] = df["Patient ID"].astype(str).str.strip()
    df[slice_col] = pd.to_numeric(df[slice_col], errors="coerce")

    print("Unique patients:", df["Patient ID"].nunique())
    print("Missing slice numbers:", int(df[slice_col].isna().sum()))
    print(
        "Duplicate patient/slice pairs:",
        int(df.duplicated(["Patient ID", slice_col]).sum())
    )
    print(
        "Duplicate filenames:",
        int(df["File name"].astype(str).duplicated().sum())
    )

    print("\nExample rows:")
    print(df[["Patient ID", "File name", slice_col]].head(5).to_string(index=False))
