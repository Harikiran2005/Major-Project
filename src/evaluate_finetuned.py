import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from model import SliceTransformer
from finetune_cnn import FineTuneCNNTransformer


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

BATCH_SIZE = 8

MODEL_PATH = "outputs/best_finetuned_pilot.pth"
TEST_CSV = "data/splits/test_final.csv"

COVID_DATA_DIR = "data/covid_25d"
CANCER_DATA_DIR = "data/cancer_25d"

CLASS_NAMES = [
    "Normal",
    "COVID",
    "CAP",
    "Lung Cancer"
]


# ============================================================
# LOAD TEST DATA
# ============================================================

from dataset import LungDiseaseDataset

print("=" * 60)
print("FINE-TUNED MODEL TEST EVALUATION")
print("=" * 60)

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

print("\nTest samples:", len(test_dataset))


# ============================================================
# LOAD MODEL
# ============================================================

model = FineTuneCNNTransformer()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

print("\n✓ Fine-tuned pilot model loaded")


# ============================================================
# PREDICTIONS
# ============================================================

all_predictions = []
all_labels = []

print("\nRunning test evaluation...")

with torch.no_grad():

    for batch_idx, (images, labels) in enumerate(
        test_loader
    ):

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )

        if (batch_idx + 1) % 50 == 0:

            print(
                f"Processed "
                f"{batch_idx + 1}/"
                f"{len(test_loader)} batches"
            )


all_predictions = np.array(
    all_predictions
)

all_labels = np.array(
    all_labels
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


print("\n" + "=" * 60)
print("FINE-TUNED PILOT TEST RESULTS")
print("=" * 60)

print(
    f"\nTest Accuracy : {accuracy:.4f}"
)

print(
    f"Test Macro F1 : {macro_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# COMPARISON WITH BASELINE
# ============================================================

BASELINE_ACCURACY = 0.7758
BASELINE_F1 = 0.8070

print("\n" + "=" * 60)
print("BASELINE vs FINE-TUNED PILOT")
print("=" * 60)

print(
    f"\nBaseline Accuracy : "
    f"{BASELINE_ACCURACY:.4f}"
)

print(
    f"Fine-tuned Accuracy: "
    f"{accuracy:.4f}"
)

print(
    f"\nBaseline Macro F1 : "
    f"{BASELINE_F1:.4f}"
)

print(
    f"Fine-tuned Macro F1: "
    f"{macro_f1:.4f}"
)

accuracy_change = (
    accuracy - BASELINE_ACCURACY
)

f1_change = (
    macro_f1 - BASELINE_F1
)

print(
    f"\nAccuracy change : "
    f"{accuracy_change:+.4f}"
)

print(
    f"Macro F1 change : "
    f"{f1_change:+.4f}"
)


# ============================================================
# DECISION
# ============================================================

if macro_f1 > BASELINE_F1:

    print(
        "\n✓ Fine-tuned pilot improved "
        "Macro F1."
    )

else:

    print(
        "\n✗ Fine-tuned pilot did not "
        "improve Macro F1."
    )

print("\n✓ Evaluation completed.")