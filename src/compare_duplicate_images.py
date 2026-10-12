
import os
import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(ROOT, "..", "meta_data_normal.csv")
IMAGE_ROOT = os.path.join(
    ROOT, "..", "curated_data", "curated_data", "1NonCOVID"
)

df = pd.read_csv(META, encoding="latin1")
df["Slice"] = pd.to_numeric(df["Slice"], errors="coerce")
df["Patient ID"] = df["Patient ID"].astype(str).str.strip()

duplicates = df[
    df.duplicated(["Patient ID", "Slice"], keep=False)
].sort_values(["Patient ID", "Slice", "File name"])

def find_image(filename):
    for current_root, _, files in os.walk(IMAGE_ROOT):
        if filename in files:
            return os.path.join(current_root, filename)
    return None

checked = 0
identical = 0
different = 0
missing = 0

for (patient, slice_no), group in duplicates.groupby(
    ["Patient ID", "Slice"], sort=True
):
    rows = list(group["File name"])

    if len(rows) < 2:
        continue

    path_a = find_image(rows[0])
    path_b = find_image(rows[1])

    if not path_a or not path_b:
        missing += 1
        continue

    a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)

    if a is None or b is None or a.shape != b.shape:
        different += 1
        continue

    checked += 1

    if np.array_equal(a, b):
        identical += 1
    else:
        different += 1

    if checked <= 10:
        print(
            patient, slice_no,
            "IDENTICAL" if np.array_equal(a, b) else "DIFFERENT",
            rows[0], "|", rows[1]
        )

print("\n--- SUMMARY ---")
print("Pairs compared:", checked)
print("Pixel-identical pairs:", identical)
print("Different pairs:", different)
print("Pairs with missing/unreadable files:", missing)
