import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, classification_report

from model import SliceTransformer


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

BATCH_SIZE = 32
EPOCHS = 20

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 4

FEATURE_DIR = "data/features"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD FEATURES
# ============================================================

def load_features(split):

    path = os.path.join(
        FEATURE_DIR,
        f"{split}_features.pt"
    )

    data = torch.load(
        path,
        map_location="cpu"
    )

    features = data["features"]
    labels = data["labels"]

    print(
        f"{split.upper()}: "
        f"features={features.shape}, "
        f"labels={labels.shape}"
    )

    return features, labels


train_features, train_labels = load_features("train")
val_features, val_labels = load_features("val")
test_features, test_labels = load_features("test")


# ============================================================
# DATASETS
# ============================================================

train_dataset = TensorDataset(
    train_features,
    train_labels
)

val_dataset = TensorDataset(
    val_features,
    val_labels
)

test_dataset = TensorDataset(
    test_features,
    test_labels
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_counts = torch.bincount(
    train_labels,
    minlength=NUM_CLASSES
).float()

print("\nClass counts:")
print(class_counts)

print("\nClass mapping:")
print("0 = Normal")
print("1 = COVID")
print("2 = CAP")
print("3 = Lung Cancer")


# ============================================================
# BALANCED SAMPLING
# ============================================================

# Give each class an inverse-frequency weight.
# Minority classes receive larger sampling probability.

class_weights = 1.0 / class_counts

sample_weights = class_weights[
    train_labels
]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_labels),
    replacement=True
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

model = SliceTransformer(
    feature_dim=512,
    transformer_dim=128,
    num_heads=4,
    num_layers=2,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

# IMPORTANT:
# We use normal CrossEntropyLoss because the sampler
# already handles class imbalance.

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# VALIDATION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(features)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    avg_loss = total_loss / len(loader)

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

    return avg_loss, accuracy, macro_f1


# ============================================================
# TRAINING
# ============================================================

best_f1 = 0.0
patience = 5
epochs_without_improvement = 0

best_model_path = (
    "outputs/best_transformer_balanced.pth"
)

print("\n" + "=" * 60)
print("BALANCED TRANSFORMER TRAINING")
print("=" * 60)

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for features, labels in train_loader:

        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    train_loss = (
        running_loss /
        len(train_loader)
    )

    # Validation
    val_loss, val_accuracy, val_f1 = evaluate(
        model,
        val_loader
    )

    scheduler.step(val_f1)

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: {val_accuracy:.4f}"
    )

    print(
        f"Val Macro F1: {val_f1:.4f}"
    )

    print(
        f"Learning Rate: "
        f"{optimizer.param_groups[0]['lr']:.6f}"
    )

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_f1 > best_f1:

        best_f1 = val_f1

        torch.save(
            model.state_dict(),
            best_model_path
        )

        epochs_without_improvement = 0

        print("✓ Best balanced model saved")

    else:

        epochs_without_improvement += 1

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if epochs_without_improvement >= patience:

        print(
            "\nEarly stopping triggered."
        )

        break


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("BALANCED MODEL - FINAL TEST")
print("=" * 60)

model.load_state_dict(
    torch.load(
        best_model_path,
        map_location=DEVICE
    )
)

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(DEVICE)

        outputs = model(features)

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


# ============================================================
# TEST METRICS
# ============================================================

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

test_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

print(
    f"\nTest Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Macro F1: "
    f"{test_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "Normal",
            "COVID",
            "CAP",
            "Lung Cancer"
        ],
        zero_division=0,
        digits=4
    )
)

print("\n✓ Balanced training completed.")