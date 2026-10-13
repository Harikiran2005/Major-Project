
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = ROOT / "data" / "splits"
COVID_DIR = ROOT / "data" / "covid_25d_v2"
CANCER_DIR = ROOT / "data" / "cancer_25d"

SPLITS = ["train", "val", "test"]

CLASS_FOLDERS = {
    0: "normal",
    1: "covid",
    2: "cap",
    3: "cancer",
}

# --------------------------------------------------
# INDEX PREPROCESSED FILES
# --------------------------------------------------

print("Indexing preprocessed CT windows...")

file_index = {}

for label, class_name in CLASS_FOLDERS.items():
    folder = CANCER_DIR if label == 3 else COVID_DIR / class_name

    if not folder.exists():
        print(f"WARNING: Missing directory: {folder}")
        file_index[label] = []
        continue

    files = sorted(folder.glob("*.npy"))
    file_index[label] = files

    print(f"{class_name}: {len(files)} files")


# --------------------------------------------------
# AUDIT PATIENT MAPPING
# --------------------------------------------------

all_patient_sets = {}
all_window_sets = {}

for split in SPLITS:
    csv_path = SPLIT_DIR / f"{split}_final.csv"

    print("\n" + "=" * 65)
    print(f"AUDITING {split.upper()} SPLIT")
    print("=" * 65)

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    required = {"patient_id", "class_name", "label"}
    if not required.issubset(df.columns):
        print(f"ERROR: Missing columns: {required - set(df.columns)}")
        continue

    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["class_name"] = df["class_name"].astype(str).str.strip().str.lower()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    patients = set()
    matched_patients = set()
    unmatched = []
    matched_windows = set()
    class_mismatches = []

    for _, row in df.iterrows():
        patient_id = row["patient_id"]
        label_value = row["label"]

        if pd.isna(label_value):
            unmatched.append((patient_id, "invalid label"))
            continue

        label = int(label_value)
        patients.add(patient_id)

        if label not in CLASS_FOLDERS:
            unmatched.append((patient_id, f"unknown label {label}"))
            continue

        # Cancer files use the patient_id_25d.npy naming convention.
        if label == 3:
            expected_file = CANCER_DIR / f"{patient_id}_25d.npy"

            if expected_file.exists():
                matches = [expected_file]
            else:
                matches = list(CANCER_DIR.glob(f"{patient_id}_*.npy"))
        else:
            class_dir = COVID_DIR / CLASS_FOLDERS[label]
            matches = list(class_dir.glob(f"{patient_id}_*.npy"))

        if not matches:
            unmatched.append((patient_id, CLASS_FOLDERS[label]))
            continue

        matched_patients.add(patient_id)

        for path in matches:
            matched_windows.add(str(path.resolve()))

        # Check whether the same prefix occurs in another class directory.
        if label != 3:
            for other_label, other_name in CLASS_FOLDERS.items():
                if other_label in (label, 3):
                    continue

                other_dir = COVID_DIR / other_name
                if other_dir.exists():
                    other_matches = list(
                        other_dir.glob(f"{patient_id}_*.npy")
                    )
                    if other_matches:
                        class_mismatches.append(
                            (patient_id, CLASS_FOLDERS[label], other_name)
                        )

    all_patient_sets[split] = patients
    all_window_sets[split] = matched_windows

    print(f"CSV patient rows: {len(df)}")
    print(f"Unique patient IDs: {len(patients)}")
    print(f"Patients with filename matches: {len(matched_patients)}")
    print(f"Patients without filename matches: {len(unmatched)}")
    print(f"Matched windows: {len(matched_windows)}")

    if unmatched:
        print("\nFirst 20 unmatched patients:")
        for patient_id, reason in unmatched[:20]:
            print(f"  {patient_id} | expected class: {reason}")

    if class_mismatches:
        print("\nWARNING: Matching prefixes found in other classes:")
        for patient_id, expected, other in class_mismatches[:20]:
            print(f"  {patient_id}: expected {expected}, also found in {other}")


# --------------------------------------------------
# CHECK SPLIT OVERLAP
# --------------------------------------------------

print("\n" + "=" * 65)
print("SPLIT OVERLAP CHECK")
print("=" * 65)

for i, split_a in enumerate(SPLITS):
    for split_b in SPLITS[i + 1:]:
        if split_a not in all_patient_sets or split_b not in all_patient_sets:
            continue

        shared_patients = (
            all_patient_sets[split_a] & all_patient_sets[split_b]
        )

        shared_windows = (
            all_window_sets[split_a] & all_window_sets[split_b]
        )

        print(f"\n{split_a} vs {split_b}")
        print(f"Shared patient IDs: {len(shared_patients)}")
        print(f"Shared exact file paths: {len(shared_windows)}")

        if shared_patients:
            print("Example shared IDs:", list(shared_patients)[:10])

print("\nAudit completed. No files were modified.")
