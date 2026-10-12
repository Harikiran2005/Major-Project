
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "covid_25d_v2"

files = list(DATA_ROOT.rglob("*.npy"))

print("Total windows:", len(files))

for class_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
    count = len(list(class_dir.glob("*.npy")))
    print(f"{class_dir.name}: {count} windows")

shape_counts = {}
invalid = []

for file in files:
    try:
        arr = np.load(file, allow_pickle=False)

        shape_counts[str(arr.shape)] = (
            shape_counts.get(str(arr.shape), 0) + 1
        )

        if (
            arr.shape != (3, 224, 224)
            or arr.dtype != np.float32
            or not np.isfinite(arr).all()
            or arr.min() < 0
            or arr.max() > 1
        ):
            invalid.append(str(file))

    except Exception as exc:
        invalid.append(f"{file}: {exc}")

print("\nShape counts:", shape_counts)
print("Invalid windows:", len(invalid))

if invalid:
    print("First five problems:")
    for item in invalid[:5]:
        print(item)
