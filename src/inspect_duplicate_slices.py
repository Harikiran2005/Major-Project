
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASETS = [
    ("normal", "meta_data_normal.csv", "Slice"),
    ("covid", "meta_data_covid.csv", "Slice"),
    ("cap", "meta_data_cap.csv", "Slices_x"),
]

for name, filename, slice_col in DATASETS:
    path = os.path.join(ROOT, "..", filename)
    df = pd.read_csv(path, encoding="latin1")

    df[slice_col] = pd.to_numeric(df[slice_col], errors="coerce")
    duplicates = df[
        df.duplicated(["Patient ID", slice_col], keep=False)
    ].sort_values(["Patient ID", slice_col, "File name"])

    print(f"\n{'=' * 60}\n{name.upper()} DUPLICATES")
    print("Rows involved:", len(duplicates))

    if not duplicates.empty:
        print(
            duplicates[["Patient ID", slice_col, "File name"]]
            .head(20)
            .to_string(index=False)
        )

    if name == "covid":
        missing = df[df[slice_col].isna()]
        print("\nCOVID rows with missing slice numbers:", len(missing))
        print(
            missing[["Patient ID", "File name"]]
            .head(10)
            .to_string(index=False)
        )
