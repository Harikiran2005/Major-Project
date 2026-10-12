
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import LungDiseaseDataset
from model import SliceTransformer
from finetune_cnn import FineTuneCNNTransformer


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_CSV = os.path.join(BASE_DIR, "data", "splits", "test_final.csv")
COVID_DATA_DIR = os.path.join(BASE_DIR, "data", "covid_25d")
CANCER_DATA_DIR = os.path.join(BASE_DIR, "data", "cancer_25d")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "best_finetuned_full.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

CLASS_NAMES = ["Normal", "COVID", "CAP", "Lung Cancer"]
BATCH_SIZE = 8


def main():
    print("=" * 55)
    print("FULL FINE-TUNED MODEL EVALUATION")
    print("=" * 55)

    device = torch.device("cpu")

    test_dataset = LungDiseaseDataset(
        TEST_CSV,
        COVID_DATA_DIR,
        CANCER_DATA_DIR
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = FineTuneCNNTransformer().to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.eval()

    print("\nModel loaded:", MODEL_PATH)
    print("Test samples:", len(test_dataset))

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_loader, start=1):
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

            if batch_index % 50 == 0:
                print(f"Processed {batch_index}/{len(test_loader)} batches")

    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 55)
    print("TEST RESULTS")
    print("=" * 55)
    print(f"Test Accuracy : {accuracy:.4f}")
    print(f"Test Macro F1 : {macro_f1:.4f}")

    print("\nClassification Report:\n")
    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1, 2, 3],
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0
        )
    )

    cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2, 3])
    cm_normalized = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3],
        normalize="true"
    )

    print("Confusion Matrix:\n", cm)
    print("\nNormalized Confusion Matrix:\n", np.round(cm_normalized, 3))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for matrix, filename, title, fmt in [
        (cm, "confusion_matrix_full_finetuned.png",
         "Full Fine-Tuned Model - Confusion Matrix", "d"),
        (cm_normalized, "confusion_matrix_full_finetuned_normalized.png",
         "Full Fine-Tuned Model - Normalized Confusion Matrix", ".2f")
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=CLASS_NAMES
        )
        display.plot(
            ax=ax,
            cmap="Blues",
            values_format=fmt,
            colorbar=False
        )
        ax.set_title(title)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()

        save_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("\nSaved:", save_path)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
