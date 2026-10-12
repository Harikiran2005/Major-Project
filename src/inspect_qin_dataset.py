
from pathlib import Path
import glob
import pandas as pd
import pydicom

ROOT = Path(r"C:\Users\Harikiran\Downloads\qin_lung_ct")
OUTPUT = Path("data/qin_lung_ct_inventory.csv")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

files = glob.glob(str(ROOT / "**" / "*.dcm"), recursive=True)
records = []

for i, file_path in enumerate(files, start=1):
    try:
        d = pydicom.dcmread(
            file_path,
            stop_before_pixels=True,
            specific_tags=[
                "PatientID",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "SOPInstanceUID",
                "SeriesDescription",
                "Modality",
                "BodyPartExamined",
                "ImagePositionPatient",
                "InstanceNumber",
                "ContrastBolusAgent",
                "SliceThickness",
            ],
        )

        position = getattr(d, "ImagePositionPatient", None)

        records.append({
            "patient_id": str(getattr(d, "PatientID", "")),
            "study_uid": str(getattr(d, "StudyInstanceUID", "")),
            "series_uid": str(getattr(d, "SeriesInstanceUID", "")),
            "sop_uid": str(getattr(d, "SOPInstanceUID", "")),
            "series_description": str(getattr(d, "SeriesDescription", "")),
            "modality": str(getattr(d, "Modality", "")),
            "body_part": str(getattr(d, "BodyPartExamined", "")),
            "z_position": float(position[2]) if position and len(position) >= 3 else None,
            "instance_number": getattr(d, "InstanceNumber", None),
            "contrast": str(getattr(d, "ContrastBolusAgent", "")),
            "slice_thickness": getattr(d, "SliceThickness", None),
            "file_path": file_path,
        })

    except Exception as exc:
        print(f"Could not read {file_path}: {exc}")

    if i % 500 == 0:
        print(f"Inspected {i}/{len(files)} files")

df = pd.DataFrame(records)
df.to_csv(OUTPUT, index=False)

print("\nInventory saved to:", OUTPUT.resolve())
print("Readable DICOM files:", len(df))
print("Unique patients:", df["patient_id"].nunique())
print("Unique series:", df["series_uid"].nunique())
