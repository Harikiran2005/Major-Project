import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SPLIT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "splits"
)

os.makedirs(SPLIT_DIR, exist_ok=True)


# ============================================================
# DATASET INFORMATION
# ============================================================

DATASETS = [
    {
        "name": "normal",
        "metadata": "meta_data_normal.csv",
        "label": 0
    },
    {
        "name": "covid",
        "metadata": "meta_data_covid.csv",
        "label": 1
    },
    {
        "name": "cap",
        "metadata": "meta_data_cap.csv",
        "label": 2
    }
]


# ============================================================
# READ PATIENT IDS
# ============================================================

all_patients = []

for dataset in DATASETS:

    metadata_path = os.path.join(
        PROJECT_ROOT,
        "..",
        dataset["metadata"]
    )

    print("\nReading:", dataset["name"])
    print("File:", metadata_path)

    df = pd.read_csv(
        metadata_path,
        encoding="latin1"
    )

    # Get unique patients
    patients = (
        df["Patient ID"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    print(
        "Patients:",
        len(patients)
    )

    for patient_id in patients:

        all_patients.append({
            "patient_id": patient_id,
            "class_name": dataset["name"],
            "label": dataset["label"]
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

patients_df = pd.DataFrame(
    all_patients
)

print("\n" + "=" * 60)
print("TOTAL PATIENTS")
print("=" * 60)

print(
    patients_df.groupby(
        "class_name"
    ).size()
)


# ============================================================
# STRATIFIED TRAIN / TEMP SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    patients_df,
    test_size=0.30,
    stratify=patients_df["label"],
    random_state=42
)


# ============================================================
# VALIDATION / TEST SPLIT
# ============================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)


# ============================================================
# SAVE SPLITS
# ============================================================

train_path = os.path.join(
    SPLIT_DIR,
    "train.csv"
)

val_path = os.path.join(
    SPLIT_DIR,
    "val.csv"
)

test_path = os.path.join(
    SPLIT_DIR,
    "test.csv"
)


train_df.to_csv(
    train_path,
    index=False
)

val_df.to_csv(
    val_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PATIENT-LEVEL SPLIT COMPLETE")
print("=" * 60)

print(
    f"\nTrain patients: {len(train_df)}"
)

print(
    f"Validation patients: {len(val_df)}"
)

print(
    f"Test patients: {len(test_df)}"
)


print("\nTRAIN DISTRIBUTION")
print(
    train_df.groupby(
        "class_name"
    ).size()
)


print("\nVALIDATION DISTRIBUTION")
print(
    val_df.groupby(
        "class_name"
    ).size()
)


print("\nTEST DISTRIBUTION")
print(
    test_df.groupby(
        "class_name"
    ).size()
)


print("\nFiles created:")

print(train_path)
print(val_path)
print(test_path)