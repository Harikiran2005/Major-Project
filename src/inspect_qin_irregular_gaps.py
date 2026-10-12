
from pathlib import Path
import pydicom
import numpy as np

ROOT = Path(r"C:\Users\Harikiran\Downloads\qin_lung_ct")
TARGETS = {"R0019", "R0098"}

for patient in sorted(TARGETS):
    rows = []

    for path in ROOT.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)

            if ds.get("PatientID") != patient:
                continue
            if "ImagePositionPatient" not in ds:
                continue

            rows.append({
                "z": float(ds.ImagePositionPatient[2]),
                "instance": ds.get("InstanceNumber", "?"),
                "path": path.name,
            })
        except Exception:
            continue

    rows.sort(key=lambda x: x["z"])
    print(f"\n--- {patient}: {len(rows)} slices ---")

    for i in range(1, len(rows)):
        gap = rows[i]["z"] - rows[i - 1]["z"]

        if gap < 0.1 or gap > 6.0:
            print(
                f"Previous: z={rows[i-1]['z']:.3f}, "
                f"instance={rows[i-1]['instance']}, "
                f"file={rows[i-1]['path']}"
            )
            print(
                f"Current:  z={rows[i]['z']:.3f}, "
                f"instance={rows[i]['instance']}, "
                f"file={rows[i]['path']}"
            )
            print(f"Gap: {gap:.3f} mm")
