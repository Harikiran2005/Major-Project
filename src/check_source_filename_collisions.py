
import os
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_ROOT = os.path.normpath(
    os.path.join(
        PROJECT_ROOT,
        "..",
        "curated_data",
        "curated_data",
    )
)

FOLDERS = ["1NonCOVID", "2COVID", "3CAP"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

for folder in FOLDERS:
    folder_path = os.path.join(DATA_ROOT, folder)
    files_by_name = defaultdict(list)

    print(f"\nChecking {folder}...")

    if not os.path.isdir(folder_path):
        print("Folder not found:", folder_path)
        continue

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                files_by_name[filename].append(
                    os.path.join(root, filename)
                )

    collisions = {
        name: paths
        for name, paths in files_by_name.items()
        if len(paths) > 1
    }

    print("Unique image filenames:", len(files_by_name))
    print("Duplicate basenames:", len(collisions))

    for name, paths in list(collisions.items())[:10]:
        print(f"\n{name}")
        for path in paths:
            print("  ", path)
