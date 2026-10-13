
import os
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPLITS = {
    "train": os.path.join(ROOT, "data", "splits", "train_final.csv"),
    "val": os.path.join(ROOT, "data", "splits", "val_final.csv"),
    "test": os.path.join(ROOT, "data", "splits", "test_final.csv"),
}

COVID_DIR = os.path.join(ROOT, "data", "covid_25d_v2")
CANCER_DIR = os.path.join(ROOT, "data", "cancer_25d")


def resolve_files(csv_path):
    df = pd.read_csv(csv_path)
    files = []

    for _, row in df.iterrows():
        patient = str(row["patient_id"]).strip()
        class_name = str(row["class_name"]).strip().lower()

        # Match the resolution logic in dataset.py
        explicit = str(row.get("filepath", "")).strip()

        if explicit and explicit.lower() not in {"nan", "none"}:
            path = os.path.normpath(explicit)
            if os.path.isfile(path):
                files.append(path)
            continue

        if class_name == "cancer":
            path = os.path.join(CANCER_DIR, f"{patient}_25d.npy")
            if os.path.isfile(path):
                files.append(path)
            continue

        class_dir = os.path.join(COVID_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue

        prefix = patient + "_"
        for filename in os.listdir(class_dir):
            if filename.endswith(".npy") and filename.startswith(prefix):
                files.append(os.path.join(class_dir, filename))

    return sorted(set(os.path.abspath(p) for p in files))


def content_hash(path):
    # Hash array contents rather than .npy file headers.
    array = np.load(path, mmap_mode="r", allow_pickle=False)

    if not np.issubdtype(array.dtype, np.number):
        return None

    if not np.isfinite(array).all():
        return None

    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())

    # Process in chunks to limit additional memory use.
    flat = array.reshape(-1)
    chunk_size = 1_000_000

    for start in range(0, flat.size, chunk_size):
        chunk = np.ascontiguousarray(flat[start:start + chunk_size])
        digest.update(chunk.tobytes())

    return digest.hexdigest()


def main():
    hashes = defaultdict(list)
    path_owners = defaultdict(list)
    total = 0

    for split, csv_path in SPLITS.items():
        if not os.path.isfile(csv_path):
            print(f"ERROR: Missing split CSV: {csv_path}")
            return

        files = resolve_files(csv_path)
        print(f"{split}: resolved {len(files)} unique files")

        for path in files:
            total += 1
            path_owners[path].append(split)

            try:
                digest = content_hash(path)
                if digest:
                    hashes[digest].append((split, path))
            except Exception as exc:
                print(f"WARNING: Could not read {path}: {exc}")

    cross_split_duplicates = []

    for digest, entries in hashes.items():
        splits_present = {split for split, _ in entries}

        if len(splits_present) > 1:
            cross_split_duplicates.append(entries)

    reused_paths = {
        path: sorted(set(splits))
        for path, splits in path_owners.items()
        if len(set(splits)) > 1
    }

    print("\n========== AUDIT SUMMARY ==========")
    print("Total resolved file references:", total)
    print("Unique array hashes:", len(hashes))
    print("Exact-content duplicate groups crossing splits:",
          len(cross_split_duplicates))
    print("Same file paths reused across splits:", len(reused_paths))

    report = []

    for group_number, entries in enumerate(cross_split_duplicates, 1):
        print(f"\nDuplicate group {group_number}:")
        for split, path in entries:
            print(f"  [{split}] {path}")
            report.append({
                "group": group_number,
                "split": split,
                "filepath": path,
            })

    output = os.path.join(ROOT, "outputs", "cross_split_duplicates.csv")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    pd.DataFrame(
        report, columns=["group", "split", "filepath"]
    ).to_csv(output, index=False)

    print("\nReport saved to:", output)

    if not cross_split_duplicates and not reused_paths:
        print(
            "No exact array-content duplicates were found across splits. "
            "This does not rule out near-duplicates or hidden patient overlap."
        )


if __name__ == "__main__":
    main()
