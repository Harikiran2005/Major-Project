
import os
import pydicom

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DICOM_ROOT = r"C:\Users\Harikiran\Downloads\lungct_diagnosis"
PATIENT_ID = "R_004"
TARGET_INSTANCE = 34  # Set this if you know the tumor InstanceNumber.

patient_dir = os.path.join(DICOM_ROOT, PATIENT_ID)

if not os.path.isdir(patient_dir):
    raise FileNotFoundError(
        f"Patient directory not found: {patient_dir}"
    )

records = []

for root, _, files in os.walk(patient_dir):
    for filename in files:
        if not filename.lower().endswith(".dcm"):
            continue

        path = os.path.join(root, filename)

        try:
            ds = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=[
                    "InstanceNumber",
                    "ImagePositionPatient",
                    "SOPInstanceUID",
                    "SeriesInstanceUID",
                    "ImageOrientationPatient",
                    "SpacingBetweenSlices",
                    "SliceThickness",
                ],
            )

            instance = getattr(ds, "InstanceNumber", None)
            position = getattr(ds, "ImagePositionPatient", None)

            if instance is None or position is None:
                continue

            records.append({
                "instance": int(instance),
                "z": float(position[2]),
                "uid": str(getattr(ds, "SOPInstanceUID", "")),
                "series": str(getattr(ds, "SeriesInstanceUID", "")),
                "orientation": str(
                    getattr(ds, "ImageOrientationPatient", "")
                ),
                "spacing": getattr(ds, "SpacingBetweenSlices", None),
                "thickness": getattr(ds, "SliceThickness", None),
            })

        except Exception as exc:
            print("Could not read:", path, "|", exc)

if not records:
    raise RuntimeError(
        "No DICOM slices with InstanceNumber and ImagePositionPatient found."
    )

# Separate series: different series should not be mixed.
series_ids = sorted({record["series"] for record in records})

print("Patient:", PATIENT_ID)
print("DICOM slices indexed:", len(records))
print("Series found:", len(series_ids))

for series_id in series_ids:
    series_records = [
        record for record in records
        if record["series"] == series_id
    ]

    series_records.sort(key=lambda record: record["z"])

    print("\nSeries:", series_id)
    print("Slices in series:", len(series_records))

    for record in series_records[:10]:
        print(
            f"Instance={record['instance']}, "
            f"Z={record['z']:.3f}, "
            f"Thickness={record['thickness']}, "
            f"Spacing={record['spacing']}"
        )

    z_values = [record["z"] for record in series_records]

    if len(z_values) > 1:
        gaps = [
            round(abs(z_values[i + 1] - z_values[i]), 4)
            for i in range(len(z_values) - 1)
        ]
        print("First slice-to-slice Z gaps:", gaps[:10])
print("\nTUMOR SLICE AND NEIGHBORS")

for record in records:
    if abs(record["instance"] - TARGET_INSTANCE) <= 1:
        print(
            f"Instance={record['instance']}, "
            f"Z={record['z']:.3f}, "
            f"UID={record['uid']}"
        )