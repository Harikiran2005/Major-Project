
import os
import numpy as np

ROOTS = {
    "normal": "data/covid_25d_v2/normal",
    "covid": "data/covid_25d_v2/covid",
    "cap": "data/covid_25d_v2/cap",
    "cancer": "data/cancer_25d",
}

rng = np.random.default_rng(42)

for class_name, folder in ROOTS.items():
    if not os.path.isdir(folder):
        print(f"{class_name}: FOLDER NOT FOUND: {folder}")
        continue

    files = sorted(
        f for f in os.listdir(folder)
        if f.endswith(".npy")
    )

    if not files:
        print(f"{class_name}: no .npy files found")
        continue

    selected_indices = rng.choice(
        len(files),
        size=min(50, len(files)),
        replace=False,
    )

    sampled_values = []

    for index in selected_indices:
        path = os.path.join(folder, files[int(index)])
        array = np.load(path, mmap_mode="r")

        # Sample pixels from the middle slice.
        sampled_values.append(
            np.asarray(array[1]).ravel()[::100]
        )

    values = np.concatenate(sampled_values)

    print(
        f"{class_name}: files={len(selected_indices)}, "
        f"mean={values.mean():.4f}, "
        f"std={values.std():.4f}, "
        f"min={values.min():.4f}, "
        f"max={values.max():.4f}"
    )
