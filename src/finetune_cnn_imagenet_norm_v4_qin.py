
import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from dataset import LungDiseaseDataset
from model import SliceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 2
NUM_CLASSES = 4

COVID_DATA_DIR = "data/covid_25d_v2"
CANCER_DATA_DIR = "data/cancer_25d"

TRAIN_CSV = "data/splits/train_qin_augmented.csv"
VAL_CSV = "data/splits/val_final.csv"
TEST_CSV = "data/splits/test_final.csv"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# V3 outputs are separate from all previous experiments.
BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR, "best_finetuned_imagenet_norm_v3.pth"
)
HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history_v3.csv")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "test_results_v3.json")

CLASS_NAMES = ["Normal", "COVID", "CAP", "Lung Cancer"]


# ============================================================
# FINE-TUNABLE RESNET18
# ============================================================

class FineTuneCNN(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Freeze the CNN initially.
        for parameter in resnet.parameters():
            parameter.requires_grad = False

        # Fine-tune only the final ResNet block.
        for parameter in resnet.layer4.parameters():
            parameter.requires_grad = True

        # Remove ImageNet classification head.
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        # ImageNet normalization.
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x):
        # Input: [batch, 1, 224, 224]
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        return self.cnn(x).flatten(1)  # [batch, 512]


# ============================================================
# CNN + TRANSFORMER
# ============================================================

class FineTuneCNNTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = FineTuneCNN()

        self.transformer = SliceTransformer(
            feature_dim=512,
            transformer_dim=128,
            num_heads=4,
            num_layers=2,
            num_classes=NUM_CLASSES,
        )

    def forward(self, x):
        # Input: [batch, 3, 224, 224]
        batch_size, num_slices, height, width = x.shape

        # Apply the shared CNN to each slice.
        x = x.reshape(
            batch_size * num_slices, 1, height, width
        )
        features = self.cnn(x)

        # Restore the slice dimension.
        features = features.reshape(
            batch_size, num_slices, 512
        )

        return self.transformer(features)


# ============================================================
# DATA HELPERS
# ============================================================

def build_dataset(csv_path):
    return LungDiseaseDataset(
        csv_path,
        COVID_DATA_DIR,
        CANCER_DATA_DIR,
    )


def get_training_class_counts(dataset):
    counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)

    for _, label in dataset.samples:
        label = int(label)

        if not 0 <= label < NUM_CLASSES:
            raise ValueError(f"Unexpected class label: {label}")

        counts[label] += 1

    return counts


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, criterion):
    model.eval()

    predictions = []
    true_labels = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions.extend(
                outputs.argmax(dim=1).cpu().tolist()
            )
            true_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(1, total_samples)

    accuracy = accuracy_score(true_labels, predictions)

    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return (
        avg_loss,
        accuracy,
        macro_f1,
        predictions,
        true_labels,
    )


# ============================================================
# MAIN TRAINING ROUTINE
# ============================================================

def main():
    print("=" * 60)
    print("V3 CNN + TRANSFORMER TRAINING")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Infectious/normal data:", COVID_DATA_DIR)
    print("Cancer data:", CANCER_DATA_DIR)
    print("Checkpoint:", BEST_MODEL_PATH)

    for csv_path in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Split file not found: {csv_path}"
            )

    # Load datasets.
    train_dataset = build_dataset(TRAIN_CSV)
    val_dataset = build_dataset(VAL_CSV)
    test_dataset = build_dataset(TEST_CSV)

    if (
        len(train_dataset) == 0
        or len(val_dataset) == 0
        or len(test_dataset) == 0
    ):
        raise ValueError("At least one dataset split is empty.")

    print("\nLoaded samples:")
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    # Calculate class weights using training samples only.
    class_counts = get_training_class_counts(train_dataset)

    print("\nTraining window counts:")
    for index, name in enumerate(CLASS_NAMES):
        print(
            f"{index} - {name}: "
            f"{int(class_counts[index].item())}"
        )

    if torch.any(class_counts == 0):
        raise ValueError(
            f"A class has no training samples: {class_counts.tolist()}"
        )

    class_weights = (
        class_counts.sum() / (NUM_CLASSES * class_counts)
    )

    print(
        "Class weights:",
        [round(x, 4) for x in class_weights.tolist()],
    )

    # Data loaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Model and optimization.
    model = FineTuneCNNTransformer().to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print("\nStarting V3 training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        samples_seen = 0

        for batch_index, (images, labels) in enumerate(
            train_loader, start=1
        ):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            samples_seen += batch_size

            if batch_index % 100 == 0:
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Batch {batch_index}/{len(train_loader)}"
                )

        train_loss = running_loss / max(1, samples_seen)

        (
            val_loss,
            val_accuracy,
            val_f1,
            _,
            _,
        ) = evaluate(model, val_loader, criterion)

        scheduler.step(val_f1)

        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_f1,
            "learning_rate": current_lr,
        })

        # Save epoch history independently.
        pd.DataFrame(history).to_csv(
            HISTORY_PATH, index=False
        )

        print("\n" + "-" * 60)
        print(f"Epoch: {epoch}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Validation macro F1: {val_f1:.4f}")
        print(f"Learning rate: {current_lr:.8g}")

        # Save the best checkpoint by validation macro F1.
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                    "seed": SEED,
                    "class_names": CLASS_NAMES,
                },
                BEST_MODEL_PATH,
            )

            print("Saved new best V3 checkpoint.")

        else:
            epochs_without_improvement += 1

            print(
                "No validation macro-F1 improvement "
                f"({epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE})."
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    if not os.path.isfile(BEST_MODEL_PATH):
        raise FileNotFoundError("No V3 checkpoint was saved.")

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation macro F1: {best_val_f1:.4f}")

    # Evaluate the test set once for this experiment.
    (
        test_loss,
        test_accuracy,
        test_macro_f1,
        predictions,
        true_labels,
    ) = evaluate(model, test_loader, criterion)

    report = classification_report(
        true_labels,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1, 2, 3],
    )

    print("\n" + "=" * 60)
    print("V3 TEST RESULTS — WINDOW-LEVEL")
    print("=" * 60)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test macro F1: {test_macro_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            true_labels,
            predictions,
            labels=[0, 1, 2, 3],
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)

    # Save final metrics separately.
    results = {
        "experiment": "V3",
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "evaluation_unit": "window-level; not patient-level",
        "class_names": CLASS_NAMES,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nV3 experiment completed.")
    print("Checkpoint:", BEST_MODEL_PATH)
    print("Training history:", HISTORY_PATH)
    print("Test results:", RESULTS_PATH)


if __name__ == "__main__":
    main()


