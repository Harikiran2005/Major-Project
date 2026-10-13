
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = ROOT / "data" / "splits"
COVID_DIR = ROOT / "data" / "covid_25d_v2"
CANCER_DIR = ROOT / "data" / "cancer_25d"

CLASS_FOLDERS = {
    0: ("normal", COVID_DIR / "normal"),
    1: ("covid", COVID_DIR / "covid"),
    2: ("cap", COVID_DIR / "cap"),
    3: ("cancer", CANCER_DIR),
}

# Build a filename inventory without modifying any files.
inventory = {}

for label, (class_name, folder) in CLASS_FOLDERS.items():
    files = sorted(folder.glob("*.npy")) if folder.exists() else []
    inventory[label] = files

    print(f"\n{class_name.upper()}: {len(files)} files")
    print("Example filenames:")

    for path in files[:12]:
        print(" ", path.name)

print("\n" + "=" * 65)
print("MANIFEST ID INSPECTION")
print("=" * 65)

for split in ("train", "val", "test"):
    csv_path = SPLIT_DIR / f"{split}_final.csv"

    if not csv_path.exists():
        print(f"\nMissing CSV: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    print(f"\n{split.upper()}")

    for label, (class_name, folder) in CLASS_FOLDERS.items():
        subset = df[pd.to_numeric(df["label"], errors="coerce") == label]

        if subset.empty:
            continue

        print(f"\n  {class_name}: {len(subset)} manifest rows")

        # Display unmatched IDs using the same direct-prefix convention.
        shown = 0

        for patient_id in subset["patient_id"].astype(str).str.strip():
            if label == 3:
                matches = list(folder.glob(f"{patient_id}_*.npy"))
                exact = folder / f"{patient_id}_25d.npy"

                if exact.exists():
                    matches = [exact]
            else:
                matches = list(folder.glob(f"{patient_id}_*.npy"))

            if not matches:
                print(f"    UNMATCHED: {patient_id}")
                shown += 1

            if shown >= 10:
                break

        if shown == 0:
            print("    No unmatched IDs among inspected rows.")
