
from pathlib import Path
import pydicom
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Harikiran\Downloads\qin_lung_ct")
OUTPUT = Path("data/qin_slice_validation.csv")

records = []
errors = 0

for path in ROOT.rglob("*.dcm"):
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)

        if ds.get("Modality") != "CT":
            continue

        position = ds.get("ImagePositionPatient")
        if position is None:
            continue

        records.append({
            "patient_id": str(ds.get("PatientID", "UNKNOWN")),
            "series_uid": str(ds.get("SeriesInstanceUID", "")),
            "instance": ds.get("InstanceNumber"),
            "z": float(position[2]),
            "path": str(path),
        })

    except Exception:
        errors += 1

df = pd.DataFrame(records)
summary = []

for (patient, series), group in df.groupby(
    ["patient_id", "series_uid"]
):
    group = group.sort_values("z")
    z = group["z"].to_numpy()
    gaps = np.abs(np.diff(z))
    median_gap = float(np.median(gaps)) if len(gaps) else np.nan

    summary.append({
        "patient_id": patient,
        "series_uid": series,
        "slices": len(group),
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "median_spacing_mm": median_gap,
        "min_spacing_mm": float(gaps.min()) if len(gaps) else np.nan,
        "max_spacing_mm": float(gaps.max()) if len(gaps) else np.nan,
        "irregular_gaps": int(
            np.sum(np.abs(gaps - median_gap) > 0.1)
        ) if len(gaps) else 0,
    })

result = pd.DataFrame(summary)
result.to_csv(OUTPUT, index=False)

print("Validation report saved to:", OUTPUT.resolve())
print("Patients:", df["patient_id"].nunique())
print("Series:", len(result))
print("Slices inspected:", len(df))
print("Files with read errors:", errors)
print("Series with irregular gaps:",
      int((result["irregular_gaps"] > 0).sum()))
print()
print(result[[
    "patient_id", "slices", "median_spacing_mm",
    "min_spacing_mm", "max_spacing_mm", "irregular_gaps"
]].to_string(index=False))
