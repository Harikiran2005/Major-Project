
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

for name, filename, slice_column in DATASETS:
    path = os.path.join(PROJECT_ROOT, "..", filename)

    df = pd.read_csv(path, encoding="latin1")
    df["Patient ID"] = df["Patient ID"].astype(str).str.strip()
    df[slice_column] = pd.to_numeric(
        df[slice_column], errors="coerce"
    )
    df = df.dropna(subset=["Patient ID", slice_column])

    gaps = 0
    duplicates = 0
    adjacent_pairs = 0
    total_pairs = 0

    for _, group in df.groupby("Patient ID"):
        nums = sorted(group[slice_column].tolist())

        if len(nums) < 2:
            continue

        for a, b in zip(nums, nums[1:]):
            total_pairs += 1
            difference = b - a

            if difference == 1:
                adjacent_pairs += 1
            elif difference == 0:
                duplicates += 1
            else:
                gaps += 1

    print(f"\n{name.upper()}")
    print("Metadata rows:", len(df))
    print("Patient groups:", df["Patient ID"].nunique())
    print("Pairs checked:", total_pairs)
    print("Pairs with difference exactly 1:", adjacent_pairs)
    print("Pairs with duplicate slice numbers:", duplicates)
    print("Pairs with difference greater than 1:", gaps)
    if total_pairs:
        print(
            "Consecutive-number percentage:",
            round(100 * adjacent_pairs / total_pairs, 2),
            "%"
        )
