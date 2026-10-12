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
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

METADATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "Representative-Tumor-Slices.xlsx"
)

DICOM_ROOT = r"C:\Users\Harikiran\Downloads\lungct_diagnosis"

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "cancer_25d"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

IMAGE_SIZE = 224

WINDOW = 3


# ============================================================
# READ REPRESENTATIVE TUMOR METADATA
# ============================================================

print("\nReading tumor metadata...")

df = pd.read_excel(
    METADATA_FILE,
    header=None
)

# Row 0 = title
# Row 1 = column headers
# Row 2 onward = actual data

df = df.iloc[2:].copy()

df.columns = [
    "PatientID",
    "SOPInstanceUID",
    "InstanceNumber",
    "ImagePosition"
]

df = df.reset_index(
    drop=True
)

print(
    "Tumor records:",
    len(df)
)

print(
    "Unique patients:",
    df["PatientID"].nunique()
)


# ============================================================
# DICOM SEARCH
# ============================================================

def find_dicom_by_uid(patient_id, target_uid):
    """
    Search one patient's DICOM files and return the
    matching filename.
    """

    patient_dir = os.path.join(
        DICOM_ROOT,
        str(patient_id)
    )

    if not os.path.exists(patient_dir):

        print(
            f"Patient folder not found: {patient_id}"
        )

        return None

    for root, _, files in os.walk(patient_dir):

        for filename in files:

            if not filename.lower().endswith(".dcm"):
                continue

            path = os.path.join(
                root,
                filename
            )

            try:

                ds = pydicom.dcmread(
                    path,
                    stop_before_pixels=True,
                    specific_tags=[
                        "SOPInstanceUID"
                    ]
                )

                uid = getattr(
                    ds,
                    "SOPInstanceUID",
                    ""
                )

                if str(uid) == str(target_uid):

                    print(
                        f"Found tumor slice for "
                        f"{patient_id}: {filename}"
                    )

                    return path

            except Exception:
                continue

    print(
        f"Tumor DICOM not found for {patient_id}"
    )

    return None


# ============================================================
# READ ONE DICOM
# ============================================================

def read_image(path):

    ds = pydicom.dcmread(
        path
    )

    image = ds.pixel_array.astype(
        np.float32
    )

    # Convert to HU
    slope = float(
        getattr(
            ds,
            "RescaleSlope",
            1.0
        )
    )

    intercept = float(
        getattr(
            ds,
            "RescaleIntercept",
            0.0
        )
    )

    image = (
        image * slope
        + intercept
    )

    return image


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_ct(image):

    # Lung-oriented CT window

    lower = -1000
    upper = 400

    image = np.clip(
        image,
        lower,
        upper
    )

    image = (
        (image - lower)
        /
        (upper - lower)
    )

    return image.astype(
        np.float32
    )


# ============================================================
# RESIZE
# ============================================================

def resize_image(image):

    image_uint8 = (
        image * 255
    ).astype(
        np.uint8
    )

    pil_image = Image.fromarray(
        image_uint8
    )

    pil_image = pil_image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    return np.asarray(
        pil_image,
        dtype=np.float32
    ) / 255.0


# ============================================================
# MAIN PROCESSING
# ============================================================

print(
    "\nStarting cancer 2.5D preprocessing...\n"
)

successful = 0
failed = 0


# ============================================================
# TEMPORARY TEST
# PROCESS ONLY R_004
# ============================================================

for _, row in tqdm(
    df.iterrows(),
    total=len(df)
):

    patient_id = str(
        row["PatientID"]
    )

    tumor_uid = str(
        row["SOPInstanceUID"]
    )

    tumor_instance = int(
        float(
            row["InstanceNumber"]
        )
    )

    # --------------------------------------------------------
    # Find tumor slice
    # --------------------------------------------------------

    tumor_path = find_dicom_by_uid(
        patient_id,
        tumor_uid
    )

    if tumor_path is None:

        print(
            f"\nCould not find tumor DICOM: "
            f"{patient_id}"
        )

        failed += 1
        continue


    # --------------------------------------------------------
    # Find all DICOM slices
    # --------------------------------------------------------

    patient_dir = os.path.dirname(
        os.path.dirname(
            tumor_path
        )
    )

    slices = []


    for root, _, files in os.walk(
        patient_dir
    ):

        for filename in files:

            if not filename.lower().endswith(".dcm"):
                continue

            path = os.path.join(
                root,
                filename
            )

            try:

                ds = pydicom.dcmread(
                    path,
                    stop_before_pixels=True,
                    specific_tags=[
                        "InstanceNumber",
                        "SOPInstanceUID"
                    ]
                )

                instance = getattr(
                    ds,
                    "InstanceNumber",
                    None
                )

                if instance is not None:

                    slices.append(
                        (
                            int(instance),
                            path
                        )
                    )

            except Exception:
                continue


    if len(slices) < 3:

        print(
            f"\nNot enough slices: "
            f"{patient_id}"
        )

        failed += 1
        continue


    # --------------------------------------------------------
    # Sort slices
    # --------------------------------------------------------

    slices.sort(
        key=lambda x: x[0]
    )

    instance_to_path = {
        instance: path
        for instance, path in slices
    }


    # --------------------------------------------------------
    # Get neighboring slices
    # --------------------------------------------------------

    required_instances = [
        tumor_instance - 1,
        tumor_instance,
        tumor_instance + 1
    ]


    if not all(
        i in instance_to_path
        for i in required_instances
    ):

        print(
            f"\nMissing neighboring slice: "
            f"{patient_id}"
        )

        failed += 1
        continue


    # --------------------------------------------------------
    # Read and process 3 slices
    # --------------------------------------------------------

    processed_slices = []


    for instance in required_instances:

        image = read_image(
            instance_to_path[
                instance
            ]
        )

        image = normalize_ct(
            image
        )

        image = resize_image(
            image
        )

        processed_slices.append(
            image
        )


    # Shape:
    # (3, 224, 224)

    volume_25d = np.stack(
        processed_slices,
        axis=0
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{patient_id}_25d.npy"
    )

    np.save(
        output_file,
        volume_25d
    )

    successful += 1


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n==================================="
)

print(
    "Cancer 2.5D preprocessing complete"
)

print(
    "==================================="
)

print(
    "Successful:",
    successful
)

print(
    "Failed:",
    failed
)

print(
    "Output folder:",
    OUTPUT_DIR
)