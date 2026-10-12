
import os
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm import tqdm

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

METADATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "Representative-Tumor-Slices.xlsx",
)

DICOM_ROOT = r"C:\Users\Harikiran\Downloads\lungct_diagnosis"

# Separate output folder: original cancer data is untouched.
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "cancer_25d_multi",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 224

# Three windows containing the representative tumor slice:
# [tumor-2, tumor-1, tumor]
# [tumor-1, tumor, tumor+1]
# [tumor, tumor+1, tumor+2]
WINDOW_START_OFFSETS = [-2, -1, 0]


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def read_ct(path):
    ds = pydicom.dcmread(path)

    image = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    image = image * slope + intercept

    # Lung-oriented CT window
    image = np.clip(image, -1000, 400)
    image = (image + 1000) / 1400.0

    image = (image * 255).astype(np.uint8)

    image = Image.fromarray(image).resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    return np.asarray(image, dtype=np.float32) / 255.0


# ============================================================
# READ METADATA
# ============================================================

print("Reading representative tumor metadata...")

df = pd.read_excel(
    METADATA_FILE,
    header=None,
)

# Match the structure used in the original preprocessing script.
df = df.iloc[2:].copy()
df.columns = [
    "PatientID",
    "SOPInstanceUID",
    "InstanceNumber",
    "ImagePosition",
]
df = df.reset_index(drop=True)

print("Metadata records:", len(df))
print("Unique patients:", df["PatientID"].nunique())


# ============================================================
# INDEX EACH PATIENT'S DICOM SERIES
# ============================================================

def index_patient_series(patient_id):
    """
    Index DICOM headers and group slices by SeriesInstanceUID.
    The tumor slice and its neighbors must belong to the same series.
    """

    patient_dir = os.path.join(
        DICOM_ROOT,
        str(patient_id),
    )

    if not os.path.isdir(patient_dir):
        return {}

    series_map = {}

    for root, _, filenames in os.walk(patient_dir):
        for filename in filenames:

            if not filename.lower().endswith(".dcm"):
                continue

            path = os.path.join(root, filename)

            try:
                ds = pydicom.dcmread(
                    path,
                    stop_before_pixels=True,
                    specific_tags=[
                        "SOPInstanceUID",
                        "SeriesInstanceUID",
                        "InstanceNumber",
                        "ImagePositionPatient",
                    ],
                )

                sop_uid = str(
                    getattr(ds, "SOPInstanceUID", "")
                )

                series_uid = str(
                    getattr(ds, "SeriesInstanceUID", "")
                )

                if not sop_uid or not series_uid:
                    continue

                instance = getattr(
                    ds, "InstanceNumber", None
                )

                position = getattr(
                    ds, "ImagePositionPatient", None
                )

                z_position = None

                if position is not None and len(position) >= 3:
                    z_position = float(position[2])

                item = {
                    "path": path,
                    "sop_uid": sop_uid,
                    "instance": (
                        int(instance)
                        if instance is not None
                        else None
                    ),
                    "z": z_position,
                }

                series_map.setdefault(
                    series_uid, []
                ).append(item)

            except Exception:
                continue

    return series_map


# ============================================================
# PROCESS PATIENTS
# ============================================================

successful_windows = 0
failed_patients = 0
patients_with_windows = 0

print("\nGenerating additional cancer windows...\n")

for _, row in tqdm(df.iterrows(), total=len(df)):

    patient_id = str(row["PatientID"]).strip()
    tumor_uid = str(row["SOPInstanceUID"]).strip()

    if not patient_id or patient_id.lower() == "nan":
        failed_patients += 1
        continue

    if not tumor_uid or tumor_uid.lower() == "nan":
        failed_patients += 1
        continue

    series_map = index_patient_series(patient_id)

    # Find the exact series containing the representative tumor slice.
    target_series = None
    tumor_index = None

    for _, slices in series_map.items():

        matching_indices = [
            i for i, item in enumerate(slices)
            if item["sop_uid"] == tumor_uid
        ]

        if matching_indices:
            target_series = slices
            tumor_index = matching_indices[0]
            break

    
    if target_series is None:
        print(f"\nTumor slice not found: {patient_id}")
        print("DICOM series found:", len(series_map))

        for series_uid, items in list(series_map.items())[:3]:
            print("Series:", series_uid)
            print("Slices indexed:", len(items))
            print(
                "Example SOP UIDs:",
                [item["sop_uid"] for item in items[:3]]
            )

        failed_patients += 1
        continue


    # Sort the series physically when z positions are available.
    if all(item["z"] is not None for item in target_series):
        target_series.sort(key=lambda item: item["z"])
    elif all(item["instance"] is not None for item in target_series):
        target_series.sort(key=lambda item: item["instance"])
    else:
        print(f"\nCannot reliably order slices: {patient_id}")
        failed_patients += 1
        continue

    # Re-find the tumor after sorting.
    tumor_index = next(
        (
            i for i, item in enumerate(target_series)
            if item["sop_uid"] == tumor_uid
        ),
        None,
    )

    if tumor_index is None:
        failed_patients += 1
        continue

    patient_windows = 0

    for window_number, offset in enumerate(WINDOW_START_OFFSETS):

        start = tumor_index + offset
        end = start + 3

        if start < 0 or end > len(target_series):
            continue

        selected = target_series[start:end]

        # If instance numbers are available, reject gaps.
        instances = [item["instance"] for item in selected]
        if all(value is not None for value in instances):
            if any(instances[i + 1] - instances[i] != 1 for i in range(2)):
                continue

        if all(value is not None for value in instances):
            if any(
                instances[i + 1] - instances[i] != 1
                for i in range(2)
            ):
                continue

        # Each generated window must contain the exact tumor slice.
        if not any(
            item["sop_uid"] == tumor_uid
            for item in selected
        ):
            continue

        try:
            images = [
                read_ct(item["path"])
                for item in selected
            ]

            volume_25d = np.stack(
                images,
                axis=0,
            ).astype(np.float32)

            if volume_25d.shape != (3, 224, 224):
                continue

            output_name = (
                f"{patient_id}_win_{window_number:02d}.npy"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                output_name,
            )

            np.save(output_path, volume_25d)

            patient_windows += 1
            successful_windows += 1

        except Exception as error:
            print(
                f"\nFailed window for {patient_id}: {error}"
            )

    if patient_windows > 0:
        patients_with_windows += 1
    else:
        failed_patients += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("ADDITIONAL CANCER WINDOW PREPROCESSING COMPLETE")
print("=" * 55)
print("Metadata patient records:", df["PatientID"].nunique())
print("Patients with generated windows:", patients_with_windows)
print("Windows generated:", successful_windows)
print("Patients with no generated windows:", failed_patients)
print("Output folder:", OUTPUT_DIR)
print("\nOriginal cancer_25d files were not modified.")
