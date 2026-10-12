import os
import torch
from torch.utils.data import DataLoader

from dataset import LungDiseaseDataset
from model import CNNFeatureExtractor


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

BATCH_SIZE = 8
NUM_WORKERS = 0

FEATURE_DIR = "data/features"

# Dataset locations
COVID_DATA_DIR = "data/covid_25d"
CANCER_DATA_DIR = "data/cancer_25d"


# ============================================================
# CREATE FEATURE DIRECTORY
# ============================================================

os.makedirs(FEATURE_DIR, exist_ok=True)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(split_name):

    print("\n" + "=" * 60)
    print(f"EXTRACTING FEATURES: {split_name.upper()}")
    print("=" * 60)

    # --------------------------------------------------------
    # Split CSV
    # --------------------------------------------------------

    csv_path = f"data/splits/{split_name}_final.csv"

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = LungDiseaseDataset(
        csv_path,
        COVID_DATA_DIR,
        CANCER_DATA_DIR
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("Samples:", len(dataset))
    print("Batches:", len(loader))

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    cnn = CNNFeatureExtractor()

    cnn = cnn.to(DEVICE)
    cnn.eval()

    all_features = []
    all_labels = []

    # --------------------------------------------------------
    # Extract CNN features
    # --------------------------------------------------------

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(loader):

            images = images.to(DEVICE)

            # CNN output:
            # [B, 3, 512]

            features = cnn(images)

            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

            if (batch_idx + 1) % 50 == 0:

                print(
                    f"Processed "
                    f"{batch_idx + 1}/{len(loader)} batches"
                )

    # --------------------------------------------------------
    # Combine batches
    # --------------------------------------------------------

    all_features = torch.cat(
        all_features,
        dim=0
    )

    all_labels = torch.cat(
        all_labels,
        dim=0
    )

    print("\nFeature shape:", all_features.shape)
    print("Label shape:", all_labels.shape)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        f"{FEATURE_DIR}/{split_name}_features.pt"
    )

    torch.save(
        {
            "features": all_features,
            "labels": all_labels
        },
        output_path
    )

    print("\nSaved:")
    print(output_path)

    print("✓ Feature extraction completed")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("2.5D CNN FEATURE EXTRACTION")
    print("=" * 60)

    print("\nDevice:", DEVICE)

    extract_features("train")
    extract_features("val")
    extract_features("test")

    print("\n" + "=" * 60)
    print("ALL FEATURE EXTRACTION COMPLETED")
    print("=" * 60)