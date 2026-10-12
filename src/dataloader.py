import os
import torch
from torch.utils.data import DataLoader

from dataset import LungDiseaseDataset


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SPLIT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "splits"
)

COVID_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "covid_25d_v2"
)

CANCER_DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "cancer_25d"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 8
NUM_WORKERS = 0


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders():

    train_dataset = LungDiseaseDataset(
        os.path.join(
            SPLIT_DIR,
            "train_final.csv"
        ),
        COVID_DATA_DIR,
        CANCER_DATA_DIR
    )

    val_dataset = LungDiseaseDataset(
        os.path.join(
            SPLIT_DIR,
            "val_final.csv"
        ),
        COVID_DATA_DIR,
        CANCER_DATA_DIR
    )

    test_dataset = LungDiseaseDataset(
        os.path.join(
            SPLIT_DIR,
            "test_final.csv"
        ),
        COVID_DATA_DIR,
        CANCER_DATA_DIR
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    print("\n" + "=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    print(
        f"\nTrain batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: {len(val_loader)}"
    )

    print(
        f"Test batches: {len(test_loader)}"
    )

    # Get one batch

    images, labels = next(
        iter(train_loader)
    )

    print(
        "\nBatch image shape:",
        images.shape
    )

    print(
        "Batch label shape:",
        labels.shape
    )

    print(
        "Labels:",
        labels
    )

    print("\n✓ DataLoader working correctly")