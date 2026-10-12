import os
import pandas as pd


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


# ============================================================
# INPUT FILES
# ============================================================

train_file = os.path.join(
    SPLIT_DIR,
    "train.csv"
)

val_file = os.path.join(
    SPLIT_DIR,
    "val.csv"
)

test_file = os.path.join(
    SPLIT_DIR,
    "test.csv"
)

cancer_train_file = os.path.join(
    SPLIT_DIR,
    "cancer_train.csv"
)

cancer_val_file = os.path.join(
    SPLIT_DIR,
    "cancer_val.csv"
)

cancer_test_file = os.path.join(
    SPLIT_DIR,
    "cancer_test.csv"
)


# ============================================================
# READ FILES
# ============================================================

train_df = pd.read_csv(train_file)
val_df = pd.read_csv(val_file)
test_df = pd.read_csv(test_file)

cancer_train_df = pd.read_csv(cancer_train_file)
cancer_val_df = pd.read_csv(cancer_val_file)
cancer_test_df = pd.read_csv(cancer_test_file)


# ============================================================
# COMBINE
# ============================================================

final_train = pd.concat(
    [train_df, cancer_train_df],
    ignore_index=True
)

final_val = pd.concat(
    [val_df, cancer_val_df],
    ignore_index=True
)

final_test = pd.concat(
    [test_df, cancer_test_df],
    ignore_index=True
)


# ============================================================
# SHUFFLE EACH SPLIT
# ============================================================

final_train = final_train.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

final_val = final_val.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

final_test = final_test.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# SAVE FINAL SPLITS
# ============================================================

train_output = os.path.join(
    SPLIT_DIR,
    "train_final.csv"
)

val_output = os.path.join(
    SPLIT_DIR,
    "val_final.csv"
)

test_output = os.path.join(
    SPLIT_DIR,
    "test_final.csv"
)


final_train.to_csv(
    train_output,
    index=False
)

final_val.to_csv(
    val_output,
    index=False
)

final_test.to_csv(
    test_output,
    index=False
)


# ============================================================
# VERIFY PATIENT LEAKAGE
# ============================================================

train_patients = set(
    final_train["patient_id"]
)

val_patients = set(
    final_val["patient_id"]
)

test_patients = set(
    final_test["patient_id"]
)

train_val_overlap = (
    train_patients & val_patients
)

train_test_overlap = (
    train_patients & test_patients
)

val_test_overlap = (
    val_patients & test_patients
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL 4-CLASS PATIENT SPLIT")
print("=" * 60)

print(
    f"\nTrain patients: {len(final_train)}"
)

print(
    f"Validation patients: {len(final_val)}"
)

print(
    f"Test patients: {len(final_test)}"
)


print("\nTRAIN CLASS DISTRIBUTION")
print(
    final_train.groupby(
        "class_name"
    ).size()
)


print("\nVALIDATION CLASS DISTRIBUTION")
print(
    final_val.groupby(
        "class_name"
    ).size()
)


print("\nTEST CLASS DISTRIBUTION")
print(
    final_test.groupby(
        "class_name"
    ).size()
)


print("\n" + "=" * 60)
print("PATIENT LEAKAGE CHECK")
print("=" * 60)

print(
    "Train ∩ Validation:",
    len(train_val_overlap)
)

print(
    "Train ∩ Test:",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test:",
    len(val_test_overlap)
)


if (
    len(train_val_overlap) == 0
    and
    len(train_test_overlap) == 0
    and
    len(val_test_overlap) == 0
):

    print(
        "\n✓ NO PATIENT LEAKAGE"
    )

else:

    print(
        "\n✗ WARNING: PATIENT LEAKAGE FOUND"
    )


print("\nFiles created:")

print(train_output)
print(val_output)
print(test_output)