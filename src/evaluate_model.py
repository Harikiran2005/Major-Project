import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    f1_score
)

from model import SliceTransformer


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

BATCH_SIZE = 32

FEATURE_PATH = "data/features/test_features.pt"
MODEL_PATH = "outputs/best_transformer.pth"

CLASS_NAMES = [
    "Normal",
    "COVID",
    "CAP",
    "Lung Cancer"
]


# ============================================================
# LOAD TEST FEATURES
# ============================================================

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

data = torch.load(
    FEATURE_PATH,
    map_location="cpu"
)

features = data["features"]
labels = data["labels"]

print("\nTest features:", features.shape)
print("Test labels:", labels.shape)


# ============================================================
# DATASET
# ============================================================

dataset = TensorDataset(
    features,
    labels
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LOAD MODEL
# ============================================================

model = SliceTransformer(
    feature_dim=512,
    transformer_dim=128,
    num_heads=4,
    num_layers=2,
    num_classes=4
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

print("\n✓ Best Transformer model loaded")


# ============================================================
# PREDICTIONS
# ============================================================

all_predictions = []
all_labels = []

with torch.no_grad():

    for batch_features, batch_labels in loader:

        batch_features = batch_features.to(DEVICE)

        outputs = model(batch_features)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            batch_labels.numpy()
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
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"\nAccuracy : {accuracy:.4f}"
)

print(
    f"Macro F1 : {macro_f1:.4f}"
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
        zero_division=0,
        digits=4
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
# NORMALIZED CONFUSION MATRIX
# ============================================================

cm_normalized = (
    cm.astype(float) /
    cm.sum(axis=1, keepdims=True)
)

print("\nNormalized Confusion Matrix:")
print(
    np.round(
        cm_normalized,
        3
    )
)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

disp.plot(
    xticks_rotation=45
)

plt.title(
    "Confusion Matrix - 2.5D CNN-Transformer"
)

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# PLOT NORMALIZED CONFUSION MATRIX
# ============================================================

disp_norm = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=CLASS_NAMES
)

disp_norm.plot(
    xticks_rotation=45
)

plt.title(
    "Normalized Confusion Matrix - 2.5D CNN-Transformer"
)

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix_normalized.png",
    dpi=300
)

plt.show()


print("\n✓ Evaluation completed")

print("\nSaved:")
print("outputs/confusion_matrix.png")
print("outputs/confusion_matrix_normalized.png")