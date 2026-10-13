
import os
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from dataset import LungDiseaseDataset


class SliceTransformerReg(nn.Module):
    def __init__(
        self,
        feature_dim=1024,
        transformer_dim=128,
        num_heads=4,
        num_layers=2,
        num_classes=4,
        dropout=0.3,
    ):
        super().__init__()

        self.projection = nn.Linear(
            feature_dim, transformer_dim
        )

        self.positional_embedding = nn.Parameter(
            torch.randn(1, 3, transformer_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(
            transformer_dim, num_classes
        )

    def forward(self, features):
        x = self.projection(features)
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

# Start conservatively on the CPU-only laptop.
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4
EARLY_STOPPING_PATIENCE = 2
NUM_CLASSES = 4

COVID_DATA_DIR = "data/covid_25d_v2"
CANCER_DATA_DIR = "data/cancer_25d"

# Use the established non-QIN-augmented splits for this first run.
TRAIN_CSV = "data/splits/train_final.csv"
VAL_CSV = "data/splits/val_final.csv"
TEST_CSV = "data/splits/test_final.csv"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(
    OUTPUT_DIR, "best_densenet121_transformer_reg.pth"
)

HISTORY_PATH = os.path.join(
    OUTPUT_DIR, "training_history_densenet121_reg.csv"
)

RESULTS_PATH = os.path.join(
    OUTPUT_DIR, "test_results_densenet121_reg.json"
)

CLASS_NAMES = ["Normal", "COVID", "CAP", "Lung Cancer"]


# ============================================================
# DENSENET-121 FEATURE EXTRACTOR
# ============================================================

class DenseNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = densenet121(
            weights=DenseNet121_Weights.DEFAULT
        )

        # Freeze the entire pretrained backbone initially.
        for parameter in backbone.parameters():
            parameter.requires_grad = False

        # Fine-tune only the final dense block and final norm.
        for parameter in backbone.features.denseblock4.parameters():
            parameter.requires_grad = True

        for parameter in backbone.features.norm5.parameters():
            parameter.requires_grad = True

        # Keep convolutional features; remove ImageNet classifier.
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ImageNet normalization for replicated grayscale CT images.
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x):
        # Input: [batch, 1, height, width]
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std

        x = self.features(x)
        x = torch.relu(x)
        x = self.pool(x)

        # DenseNet-121 outputs 1024 features.
        return x.flatten(1)


# ============================================================
# DENSENET-121 + TRANSFORMER
# ============================================================

class DenseNet121Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = DenseNetFeatureExtractor()

        self.transformer = SliceTransformerReg(
    feature_dim=1024,
    transformer_dim=128,
    num_heads=4,
    num_layers=2,
    num_classes=NUM_CLASSES,
    dropout=0.3,
)

    def forward(self, x):
        # Expected input: [batch, 3, 224, 224]
        batch_size, num_slices, height, width = x.shape

        x = x.reshape(
            batch_size * num_slices, 1, height, width
        )

        features = self.cnn(x)

        # [batch * slices, 1024] -> [batch, slices, 1024]
        features = features.reshape(
            batch_size, num_slices, 1024
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


def count_labels(dataset):
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

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(1, total_samples)

    accuracy = accuracy_score(true_labels, predictions)
    macro_f1 = f1_score(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return avg_loss, accuracy, macro_f1, predictions, true_labels


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    print("=" * 68)
    print("DENSENET-121 + TRANSFORMER")
    print("=" * 68)
    print("Device:", DEVICE)

    for path in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Split file not found: {path}")

    train_dataset = build_dataset(TRAIN_CSV)
    val_dataset = build_dataset(VAL_CSV)
    test_dataset = build_dataset(TEST_CSV)

    if min(
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    ) == 0:
        raise ValueError("At least one dataset split is empty.")

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    counts = count_labels(train_dataset)

    print("\nTraining class counts:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{i} - {name}: {int(counts[i].item())}")

    if torch.any(counts == 0):
        raise ValueError(
            f"Missing training class: {counts.tolist()}"
        )

    class_weights = counts.sum() / (NUM_CLASSES * counts)

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

    # Loading pretrained weights may require an internet connection
    # the first time, unless the weights are already cached.
    model = DenseNet121Transformer().to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(DEVICE)
    )

    trainable_parameters = [
        p for p in model.parameters() if p.requires_grad
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

    print("\nStarting training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        seen = 0

        for batch_idx, (images, labels) in enumerate(
            train_loader, start=1
        ):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Batch {batch_idx}/{len(train_loader)}"
                )

        train_loss = running_loss / max(1, seen)

        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion
        )

        scheduler.step(val_f1)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        history.append(record)
        pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        print(f"Validation macro-F1: {val_f1:.4f}")

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
                    "architecture": "DenseNet121Transformer",
                },
                BEST_MODEL_PATH,
            )
            print("Saved best DenseNet checkpoint.")
        else:
            epochs_without_improvement += 1
            print(
                "No improvement: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping.")
            break

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate test split once after model selection.
    test_loss, test_acc, test_f1, preds, labels = evaluate(
        model, test_loader, criterion
    )

    report = classification_report(
        labels,
        preds,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels, preds, labels=[0, 1, 2, 3]
    )

    print("\n" + "=" * 68)
    print("DENSENET-121 TEST RESULTS — WINDOW LEVEL")
    print("=" * 68)
    print("Best epoch:", best_epoch)
    print(f"Best validation macro-F1: {best_val_f1:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test macro-F1: {test_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            labels,
            preds,
            labels=[0, 1, 2, 3],
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)

    results = {
        "architecture": "DenseNet121Transformer",
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "evaluation_unit": "window-level",
        "class_names": CLASS_NAMES,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("\nCheckpoint:", BEST_MODEL_PATH)
    print("History:", HISTORY_PATH)
    print("Results:", RESULTS_PATH)
    print("Experiment completed.")


if __name__ == "__main__":
    main()
