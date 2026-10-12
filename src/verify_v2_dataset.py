
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "covid_25d_v2"
)

for class_name in ["normal", "covid", "cap"]:
    folder = os.path.join(DATA_DIR, class_name)

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".npy")
    ]

    print(f"\n{class_name.upper()}")
    print("Files:", len(files))

    if not files:
        print("WARNING: No samples found")
        continue

    sample_path = os.path.join(folder, files[0])
    sample = np.load(sample_path)

    print("Sample:", files[0])
    print("Shape:", sample.shape)
    print("Dtype:", sample.dtype)
    print("Min:", float(sample.min()))
    print("Max:", float(sample.max()))

    assert sample.shape == (3, 224, 224)
    assert sample.dtype == np.float32
    assert np.isfinite(sample).all()
    assert sample.min() >= 0 and sample.max() <= 1

print("\nV2 sample verification complete.")
