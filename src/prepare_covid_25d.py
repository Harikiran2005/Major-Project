import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "..",
    "curated_data",
    "curated_data"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "covid_25d"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
WINDOW_SIZE = 3


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASETS = [

    {
        "name": "normal",
        "folder": "1NonCOVID",
        "metadata": "meta_data_normal.csv",
        "label": 0,
        "slice_column": "Slice"
    },

    {
        "name": "covid",
        "folder": "2COVID",
        "metadata": "meta_data_covid.csv",
        "label": 1,
        "slice_column": "Slice"
    },

    {
        "name": "cap",
        "folder": "3CAP",
        "metadata": "meta_data_cap.csv",
        "label": 2,
        "slice_column": "Slices_x"
    }

]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# IMAGE INDEX
# ============================================================

def build_image_index(folder_path):

    image_index = {}

    for root, _, files in os.walk(folder_path):

        for file in files:

            if file.lower().endswith((".png", ".jpg", ".jpeg")):

                image_index[file] = os.path.join(root, file)

    return image_index


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(config):

    dataset_name = config["name"]
    folder_name = config["folder"]
    metadata_file = config["metadata"]
    label = config["label"]
    slice_column = config["slice_column"]

    print("\n" + "=" * 60)
    print(f"Processing: {dataset_name.upper()}")
    print("=" * 60)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    image_folder = os.path.join(
        DATA_ROOT,
        folder_name
    )

    metadata_path = os.path.join(
        PROJECT_ROOT,
        "..",
        metadata_file
    )

    if not os.path.exists(image_folder):

        print(f"ERROR: Image folder not found:")
        print(image_folder)

        return

    if not os.path.exists(metadata_path):

        print(f"ERROR: Metadata file not found:")
        print(metadata_path)

        return

    # --------------------------------------------------------
    # Read metadata
    # --------------------------------------------------------

    print("Reading metadata...")

    df = pd.read_csv(
        metadata_path,
        encoding="latin1"
    )

    print(f"Metadata rows: {len(df)}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "File name",
        "Patient ID",
        slice_column
    ]

    for column in required_columns:

        if column not in df.columns:

            print(f"ERROR: Missing column: {column}")
            print("Available columns:")
            print(df.columns.tolist())

            return

    # --------------------------------------------------------
    # Build image index
    # --------------------------------------------------------

    print("Indexing images...")

    image_index = build_image_index(
        image_folder
    )

    print(
        f"Images found: {len(image_index)}"
    )

    # --------------------------------------------------------
    # Group images by patient
    # --------------------------------------------------------

    patients = {}

    for _, row in df.iterrows():

        filename = str(row["File name"]).strip()

        patient_id = str(
            row["Patient ID"]
        ).strip()

        slice_number = row[slice_column]

        if filename not in image_index:
            continue

        try:
            slice_number = float(slice_number)

        except:

            continue

        if patient_id not in patients:

            patients[patient_id] = []

        patients[patient_id].append(
            (
                slice_number,
                filename,
                image_index[filename]
            )
        )

    print(
        f"Patients found: {len(patients)}"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    dataset_output = os.path.join(
        OUTPUT_DIR,
        dataset_name
    )

    os.makedirs(
        dataset_output,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create 2.5D windows
    # --------------------------------------------------------

    total_windows = 0

    for patient_id, slices in tqdm(
        patients.items(),
        desc=f"{dataset_name} patients"
    ):

        # Sort slices by slice number

        slices.sort(
            key=lambda x: x[0]
        )

        # Need at least 3 slices

        if len(slices) < WINDOW_SIZE:

            continue

        # ----------------------------------------------------
        # Sliding 3-slice window
        # ----------------------------------------------------

        for i in range(
            len(slices) - WINDOW_SIZE + 1
        ):

            window = slices[
                i:i + WINDOW_SIZE
            ]

            processed_images = []

            valid_window = True

            # ------------------------------------------------
            # Read 3 slices
            # ------------------------------------------------

            for _, _, image_path in window:

                image = cv2.imread(
                    image_path,
                    cv2.IMREAD_GRAYSCALE
                )

                if image is None:

                    valid_window = False
                    break

                # Resize

                image = cv2.resize(
                    image,
                    (IMAGE_SIZE, IMAGE_SIZE),
                    interpolation=cv2.INTER_AREA
                )

                # Normalize [0,1]

                image = image.astype(
                    np.float32
                ) / 255.0

                processed_images.append(
                    image
                )

            if not valid_window:
                continue

            # ------------------------------------------------
            # Stack
            # ------------------------------------------------

            stack = np.stack(
                processed_images,
                axis=0
            )

            # Shape:
            # (3, 224, 224)

            # ------------------------------------------------
            # Save
            # ------------------------------------------------
            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            center_slice = window[1][0]

            if pd.isna(center_slice):
                center_slice = i + 1

            output_filename = (
                f"{patient_id}_"
                f"{int(center_slice):04d}_"
                f"{i:04d}.npy"
            )

            output_path = os.path.join(
                dataset_output,
                output_filename
            )

            np.save(
                output_path,
                stack
            )

            total_windows += 1

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("COVID / CAP / NORMAL 2.5D PREPROCESSING")
    print("=" * 60)

    print(
        f"\nDataset root:\n{DATA_ROOT}"
    )

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    for config in DATASETS:

        process_dataset(config)

    print("\n")
    print("=" * 60)
    print("ALL DATASETS COMPLETED")
    print("=" * 60)