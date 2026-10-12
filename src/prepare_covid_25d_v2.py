import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "..", "curated_data", "curated_data")

# Separate output directory: the existing data/covid_25d is not overwritten.
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "covid_25d_v2")
IMAGE_SIZE = 224

DATASETS = [
    {"name": "normal", "folder": "1NonCOVID",
     "metadata": "meta_data_normal.csv", "slice_column": "Slice"},
    {"name": "covid", "folder": "2COVID",
     "metadata": "meta_data_covid.csv", "slice_column": "Slice"},
    {"name": "cap", "folder": "3CAP",
     "metadata": "meta_data_cap.csv", "slice_column": "Slices_x"},
]


def build_image_index(folder_path):
    image_index = {}
    collisions = 0

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            if filename in image_index:
                collisions += 1
                continue
            image_index[filename] = os.path.join(root, filename)

    if collisions:
        print(f"WARNING: {collisions} duplicate image basenames found; first path retained.")
    return image_index


def read_grayscale_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(
        image, (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )
    return image.astype(np.float32) / 255.0


def process_dataset(config):
    name = config["name"]
    image_folder = os.path.join(DATA_ROOT, config["folder"])
    metadata_path = os.path.join(PROJECT_ROOT, "..", config["metadata"])
    slice_column = config["slice_column"]

    print("\n" + "=" * 60)
    print(f"PROCESSING {name.upper()} — V2")
    print("=" * 60)

    if not os.path.isdir(image_folder):
        print("ERROR: Image folder not found:", image_folder)
        return
    if not os.path.isfile(metadata_path):
        print("ERROR: Metadata file not found:", metadata_path)
        return

    df = pd.read_csv(metadata_path, encoding="latin1")
    required = ["File name", "Patient ID", slice_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        print("ERROR: Missing required columns:", missing)
        print("Available columns:", df.columns.tolist())
        return

    image_index = build_image_index(image_folder)
    print("Metadata rows:", len(df))
    print("Indexed images:", len(image_index))

    df = df.copy()
    df["File name"] = df["File name"].astype(str).str.strip()
    df["Patient ID"] = df["Patient ID"].astype(str).str.strip()
    df[slice_column] = pd.to_numeric(df[slice_column], errors="coerce")
    df = df.dropna(subset=[slice_column])
    df = df[
        (df["Patient ID"] != "")
        & (df["Patient ID"].str.lower() != "nan")
        & df["File name"].isin(image_index)
    ]

    dataset_output = os.path.join(OUTPUT_DIR, name)
    os.makedirs(dataset_output, exist_ok=True)

    saved = 0
    duplicate_rows = 0
    gap_windows = 0
    unreadable_windows = 0
    patients_processed = 0

    for patient_id, group in tqdm(
        df.groupby("Patient ID", sort=True),
        desc=f"{name} patients",
    ):
        group = group.sort_values(
            [slice_column, "File name"], kind="mergesort"
        )

        # One image per slice number; deterministic duplicate handling.
        duplicate_mask = group.duplicated(subset=[slice_column], keep="first")
        duplicate_rows += int(duplicate_mask.sum())
        group = group.loc[~duplicate_mask]

        slices = [
            (float(row[slice_column]), row["File name"], image_index[row["File name"]])
            for _, row in group.iterrows()
        ]
        if len(slices) < 3:
            continue

        patients_processed += 1
        patient_window_index = 0

        for i in range(len(slices) - 2):
            window = slices[i:i + 3]
            numbers = [item[0] for item in window]

            # Require exact consecutive slice numbers n, n+1, n+2.
            if not (
                numbers[1] - numbers[0] == 1
                and numbers[2] - numbers[1] == 1
            ):
                gap_windows += 1
                continue

            images = []
            valid = True
            for _, _, image_path in window:
                image = read_grayscale_image(image_path)
                if image is None:
                    valid = False
                    break
                images.append(image)

            if not valid:
                unreadable_windows += 1
                continue

            stack = np.stack(images, axis=0).astype(np.float32)
            center = numbers[1]
            center_label = (
                str(int(center)).zfill(4)
                if center.is_integer()
                else str(center).replace(".", "_")
            )
            output_filename = (
                f"{patient_id}_{center_label}_{patient_window_index:04d}.npy"
            )
            np.save(os.path.join(dataset_output, output_filename), stack)
            saved += 1
            patient_window_index += 1

    print(f"Patients with at least 3 unique slices: {patients_processed}")
    print(f"Duplicate slice rows skipped: {duplicate_rows}")
    print(f"Candidate windows rejected due to slice-number gaps: {gap_windows}")
    print(f"Windows skipped due to unreadable images: {unreadable_windows}")
    print(f"Valid windows saved: {saved}")
    print("Output:", dataset_output)


def main():
    print("=" * 60)
    print("COVID / CAP / NORMAL 2.5D PREPROCESSING — V2")
    print("Original data/covid_25d will NOT be overwritten.")
    print("Output:", OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for config in DATASETS:
        process_dataset(config)

    print("\nV2 preprocessing finished. Review saved-window counts before training.")


if __name__ == "__main__":
    main()
