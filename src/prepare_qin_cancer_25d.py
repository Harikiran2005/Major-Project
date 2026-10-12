
from pathlib import Path
from collections import defaultdict

import numpy as np
import pydicom
import cv2
from tqdm import tqdm


# -------------------- PATHS --------------------
INPUT_ROOT = Path(r"C:\Users\Harikiran\Downloads\qin_lung_ct")
OUTPUT_ROOT = Path("data/qin_cancer_25d")

IMAGE_SIZE = 224
WINDOW_SIZE = 3

# CT window: lung-focused intensity range
HU_MIN = -1000
HU_MAX = 400

# Skip a window if either gap is too different from
# the patient's typical slice spacing.
GAP_TOLERANCE = 0.5


def read_slice(path):
    """Read one CT slice and return its physical position and HU image."""
    ds = pydicom.dcmread(str(path))

    if ds.get("Modality") != "CT":
        return None

    if "ImagePositionPatient" not in ds:
        return None

    image = ds.pixel_array.astype(np.float32)

    slope = float(ds.get("RescaleSlope", 1))
    intercept = float(ds.get("RescaleIntercept", 0))
    image = image * slope + intercept

    z = float(ds.ImagePositionPatient[2])

    # Clip HU and normalize to [0, 1].
    image = np.clip(image, HU_MIN, HU_MAX)
    image = (image - HU_MIN) / (HU_MAX - HU_MIN)

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return {
        "patient": str(ds.get("PatientID", "UNKNOWN")),
        "series": str(ds.get("SeriesInstanceUID", "")),
        "sop": str(ds.get("SOPInstanceUID", "")),
        "z": z,
        "image": image.astype(np.float32),
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Group DICOM files by patient and series.
    grouped_paths = defaultdict(list)
    read_errors = 0

    files = list(INPUT_ROOT.rglob("*.dcm"))
    print("DICOM files discovered:", len(files))

    for path in tqdm(files, desc="Reading CT metadata"):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)

            if ds.get("Modality") != "CT":
                continue

            patient = str(ds.get("PatientID", "UNKNOWN"))
            series = str(ds.get("SeriesInstanceUID", ""))

            if patient == "UNKNOWN" or not series:
                continue

            grouped_paths[(patient, series)].append(path)

        except Exception:
            read_errors += 1

    total_windows = 0
    total_duplicates = 0
    skipped_patients = 0

    for (patient, series), paths in tqdm(
        grouped_paths.items(), desc="Preparing patients"
    ):
        slices = []

        for path in paths:
            try:
                item = read_slice(path)
                if item is not None:
                    slices.append(item)
            except Exception:
                read_errors += 1

        # Sort by physical position.
        slices.sort(key=lambda item: item["z"])

        # Deduplicate identical physical Z positions.
        unique_slices = []
        seen_z = set()

        for item in slices:
            z_key = round(item["z"], 3)

            if z_key in seen_z:
                total_duplicates += 1
                continue

            seen_z.add(z_key)
            unique_slices.append(item)

        if len(unique_slices) < WINDOW_SIZE:
            skipped_patients += 1
            continue

        z_positions = np.array(
            [item["z"] for item in unique_slices],
            dtype=np.float64
        )
        gaps = np.abs(np.diff(z_positions))
        positive_gaps = gaps[gaps > 1e-4]

        if len(positive_gaps) == 0:
            skipped_patients += 1
            continue

        typical_spacing = float(np.median(positive_gaps))

        patient_dir = OUTPUT_ROOT / patient
        patient_dir.mkdir(parents=True, exist_ok=True)

        windows_saved = 0

        for i in range(len(unique_slices) - 2):
            selected = unique_slices[i:i + WINDOW_SIZE]

            selected_gaps = np.abs(np.diff(
                [item["z"] for item in selected]
            ))

            # Do not bridge missing or irregularly spaced slices.
            if any(
                abs(gap - typical_spacing)
                > GAP_TOLERANCE * typical_spacing
                for gap in selected_gaps
            ):
                continue

            array = np.stack(
                [item["image"] for item in selected],
                axis=0
            ).astype(np.float32)

            filename = f"{patient}_{windows_saved:04d}.npy"
            np.save(patient_dir / filename, array)

            windows_saved += 1
            total_windows += 1

        if windows_saved == 0:
            skipped_patients += 1

    print("\nPreprocessing complete.")
    print("Patients/series discovered:", len(grouped_paths))
    print("Windows saved:", total_windows)
    print("Duplicate Z-position slices skipped:", total_duplicates)
    print("Patients/series with no windows:", skipped_patients)
    print("Read errors:", read_errors)
    print("Output folder:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
