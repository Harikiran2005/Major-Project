import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torchvision.models import resnet18, ResNet18_Weights

from dataset import LungDiseaseDataset
from model import SliceTransformer


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

BATCH_SIZE = 8
EPOCHS = 5

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 4

COVID_DATA_DIR = "data/covid_25d"
CANCER_DATA_DIR = "data/cancer_25d"

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FINE-TUNABLE CNN
# ============================================================

class FineTuneCNN(nn.Module):
    def __init__(self):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)

        # Freeze all CNN layers initially.
        for param in resnet.parameters():
            param.requires_grad = False

        # Fine-tune only the final ResNet block.
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        # Remove the ImageNet classifier.
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        # ImageNet normalization for the pretrained ResNet.
        # The dataset images are expected to be scaled to [0, 1].
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x):
        # Input: [B, 1, 224, 224]
        # Convert grayscale CT slices to three channels.
        x = x.repeat(1, 3, 1, 1)

        # Normalize to match the pretrained ResNet18 input convention.
        x = (x - self.mean) / self.std

        features = self.cnn(x)

        # Output: [B, 512]
        return features.flatten(1)


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
        # Input: [B, 3, 224, 224]
        batch_size, num_slices, height, width = x.shape

        # Process each slice independently through the shared CNN.
        x = x.reshape(batch_size * num_slices, 1, height, width)
        features = self.cnn(x)

        # [B * S, 512] -> [B, S, 512]
        features = features.reshape(batch_size, num_slices, 512)

        # Model inter-slice relationships with the Transformer.
        return self.transformer(features)


# ============================================================
# DATASETS
# ============================================================

print("=" * 60)
print("FULL-DATA CNN + TRANSFORMER FINE-TUNING")
print("=" * 60)

train_dataset = LungDiseaseDataset(
    "data/splits/train_final.csv",
    COVID_DATA_DIR,
    CANCER_DATA_DIR,
)

val_dataset = LungDiseaseDataset(
    "data/splits/val_final.csv",
    COVID_DATA_DIR,
    CANCER_DATA_DIR,
)

test_dataset = LungDiseaseDataset(
    "data/splits/test_final.csv",
    COVID_DATA_DIR,
    CANCER_DATA_DIR,
)

print("\nTrain samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# ============================================================
# DATALOADERS
# ============================================================

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


# ============================================================
# CLASS COUNTS AND CLASS WEIGHTS
# ============================================================

class_counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)

for _, label in train_dataset:
    class_counts[int(label)] += 1

if torch.any(class_counts == 0):
    raise ValueError(
        f"At least one class has zero training samples: {class_counts.tolist()}"
    )

print("\nClass counts:")
print(class_counts)

print("\nClass mapping:")
print("0 = Normal")
print("1 = COVID")
print("2 = CAP")
print("3 = Lung Cancer")

class_weights = class_counts.sum() / (NUM_CLASSES * class_counts)

print("\nClass weights:")
print(class_weights)


# ============================================================
# MODEL, LOSS, OPTIMIZER, SCHEDULER
# ============================================================

model = FineTuneCNNTransformer().to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

trainable_parameters = [
    parameter for parameter in model.parameters()
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


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(model, loader):
    model.eval()

    predictions = []
    labels_list = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy().tolist())
            labels_list.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader))
    accuracy = accuracy_score(labels_list, predictions)
    macro_f1 = f1_score(
        labels_list,
        predictions,
        average="macro",
        zero_division=0,
    )

    return avg_loss, accuracy, macro_f1


# ============================================================
# TRAINING
# ============================================================

best_f1 = -1.0
epochs_without_improvement = 0
PATIENCE = 2

# Use a new filename so the previously trained model is not overwritten.
best_model_path = os.path.join(
    OUTPUT_DIR,
    "best_finetuned_imagenet_norm.pth",
)

print("\n" + "=" * 60)
print("STARTING FULL FINE-TUNING WITH IMAGENET NORMALIZATION")
print("=" * 60)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_idx + 1}/{len(train_loader)}"
            )

    train_loss = running_loss / max(1, len(train_loader))
    val_loss, val_accuracy, val_f1 = evaluate(model, val_loader)
    scheduler.step(val_f1)

    print("\n" + "-" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_accuracy:.4f}")
    print(f"Val Macro F1: {val_f1:.4f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.7f}")
    print("-" * 60)

    # Save the checkpoint with the best validation Macro F1.
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), best_model_path)
        epochs_without_improvement = 0
        print("Best normalized-input model saved:", best_model_path)
    else:
        epochs_without_improvement += 1
        print(f"No improvement ({epochs_without_improvement}/{PATIENCE})")

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("NORMALIZED-INPUT MODEL - TEST")
print("=" * 60)

model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model.eval()

predictions = []
true_labels = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(DEVICE)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        predictions.extend(preds.cpu().numpy().tolist())
        true_labels.extend(labels.numpy().tolist())

        if (batch_idx + 1) % 50 == 0:
            print(f"Test batch {batch_idx + 1}/{len(test_loader)}")

test_accuracy = accuracy_score(true_labels, predictions)
test_f1 = f1_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0,
)

print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1: {test_f1:.4f}")

print("\nClassification Report:\n")
print(
    classification_report(
        true_labels,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=["Normal", "COVID", "CAP", "Lung Cancer"],
        digits=4,
        zero_division=0,
    )
)

print("\nNormalized-input experiment completed.")
print("Checkpoint saved to:", best_model_path)
