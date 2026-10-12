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
# CANCER PATIENT INFORMATION
# ============================================================

CANCER_ROOT = r"C:\Users\Harikiran\Downloads\lungct_diagnosis"

LABEL = 3
CLASS_NAME = "cancer"


# ============================================================
# FIND PATIENTS
# ============================================================

print("=" * 60)
print("CANCER PATIENT SPLIT")
print("=" * 60)

if not os.path.exists(CANCER_ROOT):

    print("\nERROR: Cancer dataset not found:")
    print(CANCER_ROOT)
    raise SystemExit


patients = []

for folder in os.listdir(CANCER_ROOT):

    folder_path = os.path.join(
        CANCER_ROOT,
        folder
    )

    if os.path.isdir(folder_path):

        patients.append(folder)


patients = sorted(
    patients
)

print(
    f"\nCancer patients found: {len(patients)}"
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

cancer_df = pd.DataFrame({
    "patient_id": patients,
    "class_name": CLASS_NAME,
    "label": LABEL
})


# ============================================================
# 70 / 15 / 15 SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    cancer_df,
    test_size=0.30,
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42
)


# ============================================================
# SAVE TEMPORARY CANCER SPLITS
# ============================================================

train_path = os.path.join(
    SPLIT_DIR,
    "cancer_train.csv"
)

val_path = os.path.join(
    SPLIT_DIR,
    "cancer_val.csv"
)

test_path = os.path.join(
    SPLIT_DIR,
    "cancer_test.csv"
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
print("CANCER SPLIT COMPLETE")
print("=" * 60)

print(
    f"\nTrain cancer patients: {len(train_df)}"
)

print(
    f"Validation cancer patients: {len(val_df)}"
)

print(
    f"Test cancer patients: {len(test_df)}"
)

print("\nFiles created:")

print(train_path)
print(val_path)
print(test_path)