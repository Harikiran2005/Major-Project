
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Run from the project root.
DEVICE = torch.device("cpu")
BATCH_SIZE = 4
NUM_CLASSES = 4
CLASS_NAMES = ["Normal", "COVID", "CAP", "Lung Cancer"]

TEST_CSV = "data/splits/test_final.csv"
COVID_DATA_DIR = "data/covid_25d_v2"
CANCER_DATA_DIR = "data/cancer_25d"
OUTPUT_DIR = "outputs"

MODELS = {
    "original": {
        "checkpoint": "outputs/best_densenet121_transformer.pth",
        "dropout": 0.2,
    },
    "regularized": {
        "checkpoint": "outputs/best_densenet121_transformer_reg.pth",
        "dropout": 0.3,
    },
}


class PatientWindowDataset(Dataset):
    """Return each CT window together with its patient ID and label."""

    def __init__(self, csv_file):
        manifest = pd.read_csv(csv_file)
        self.samples = []

        required = {"patient_id", "class_name", "label"}
        if not required.issubset(manifest.columns):
            raise ValueError(
                f"Manifest must contain columns: {sorted(required)}"
            )

        if manifest["patient_id"].isna().any():
            raise ValueError("Missing patient IDs in manifest.")

        if manifest["label"].isna().any():
            raise ValueError("Missing labels in manifest.")

        # Each patient must have exactly one ground-truth label.
        label_counts = manifest.groupby("patient_id")["label"].nunique()
        if (label_counts > 1).any():
            raise ValueError("A patient has conflicting ground-truth labels.")

        for _, row in manifest.iterrows():
            pid = str(row["patient_id"]).strip()
            class_name = str(row["class_name"]).strip().lower()
            label = int(row["label"])

            if label not in range(NUM_CLASSES):
                raise ValueError(f"Invalid label {label} for {pid}")

            paths = []

            explicit_path = str(row.get("filepath", "")).strip()
            if explicit_path.lower() not in {"", "nan", "none"}:
                if os.path.isfile(explicit_path):
                    paths = [os.path.normpath(explicit_path)]
                else:
                    raise FileNotFoundError(explicit_path)

            elif class_name == "cancer":
                path = os.path.join(
                    CANCER_DATA_DIR, f"{pid}_25d.npy"
                )
                if os.path.isfile(path):
                    paths = [path]
                else:
                    raise FileNotFoundError(
                        f"Cancer window not found for {pid}: {path}"
                    )

            else:
                folder = os.path.join(COVID_DATA_DIR, class_name)
                if not os.path.isdir(folder):
                    raise FileNotFoundError(
                        f"Class folder not found: {folder}"
                    )

                prefix = pid + "_"
                paths = sorted(
                    os.path.join(folder, name)
                    for name in os.listdir(folder)
                    if name.endswith(".npy") and name.startswith(prefix)
                )

                if not paths:
                    raise FileNotFoundError(
                        f"No CT windows found for patient {pid}"
                    )

            for path in paths:
                self.samples.append((path, label, pid))

        print(
            f"Loaded {len(self.samples)} windows from "
            f"{manifest['patient_id'].nunique()} patients."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, pid = self.samples[index]

        array = np.load(path).astype(np.float32)

        if array.shape != (3, 224, 224):
            raise ValueError(
                f"Unexpected shape {array.shape}: {path}"
            )

        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite values in {path}")

        return (
            torch.from_numpy(array),
            torch.tensor(label, dtype=torch.long),
            pid,
        )


class SliceTransformerReg(nn.Module):
    def __init__(
        self,
        feature_dim=1024,
        transformer_dim=128,
        num_heads=4,
        num_layers=2,
        num_classes=4,
        dropout=0.2,
    ):
        super().__init__()
        self.projection = nn.Linear(feature_dim, transformer_dim)
        self.positional_embedding = nn.Parameter(
            torch.randn(1, 3, transformer_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers
        )
        self.classifier = nn.Linear(transformer_dim, num_classes)

    def forward(self, features):
        x = self.projection(features)
        x = x + self.positional_embedding
        x = self.transformer(x)
        return self.classifier(x.mean(dim=1))


class DenseNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = densenet121(weights=None)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        x = self.features(x)
        x = torch.relu(x)
        return self.pool(x).flatten(1)


class DenseNet121Transformer(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.cnn = DenseNetFeatureExtractor()
        self.transformer = SliceTransformerReg(dropout=dropout)

    def forward(self, x):
        batch, slices, height, width = x.shape
        x = x.reshape(batch * slices, 1, height, width)
        features = self.cnn(x)
        features = features.reshape(batch, slices, 1024)
        return self.transformer(features)


def load_model(checkpoint_path, dropout):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    model = DenseNet121Transformer(dropout=dropout).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def evaluate_patient_level(model, loader):
    # Accumulate probability vectors for every patient.
    patient_probs = {}
    patient_labels = {}
    window_count = {}

    with torch.no_grad():
        for images, labels, patient_ids in loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            labels = labels.numpy()

            for pid, label, prob in zip(patient_ids, labels, probs):
                if pid in patient_labels and patient_labels[pid] != int(label):
                    raise ValueError(f"Conflicting labels for {pid}")

                patient_labels[pid] = int(label)
                patient_probs.setdefault(pid, []).append(prob)
                window_count[pid] = window_count.get(pid, 0) + 1

    # Patient prediction = class with highest mean window probability.
    patient_ids = sorted(patient_probs)
    y_true = []
    y_pred = []

    rows = []
    for pid in patient_ids:
        mean_prob = np.mean(patient_probs[pid], axis=0)
        pred = int(np.argmax(mean_prob))
        true = patient_labels[pid]

        y_true.append(true)
        y_pred.append(pred)

        row = {
            "patient_id": pid,
            "true_label": true,
            "predicted_label": pred,
            "num_windows": window_count[pid],
        }
        for i, name in enumerate(CLASS_NAMES):
            row[f"prob_{name.lower().replace(' ', '_')}"] = float(mean_prob[i])
        rows.append(row)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(NUM_CLASSES))
    )

    return {
        "patients": len(patient_ids),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "patient_predictions": rows,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = PatientWindowDataset(TEST_CSV)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    results = {}

    for name, config in MODELS.items():
        print("\n" + "=" * 60)
        print(f"PATIENT-LEVEL EVALUATION: {name.upper()}")
        print("=" * 60)

        model = load_model(
            config["checkpoint"],
            config["dropout"],
        )

        result = evaluate_patient_level(model, loader)
        results[name] = {
            key: value
            for key, value in result.items()
            if key != "patient_predictions"
        }

        print("Patients:", result["patients"])
        print(f"Patient-level accuracy: {result['accuracy']:.4f}")
        print(f"Patient-level macro-F1: {result['macro_f1']:.4f}")
        print("\nClassification report:")
        print(pd.DataFrame(result["classification_report"]).T.round(4))
        print("\nConfusion matrix (rows=true, columns=predicted):")
        print(np.array(result["confusion_matrix"]))

        pd.DataFrame(result["patient_predictions"]).to_csv(
            os.path.join(
                OUTPUT_DIR, f"patient_predictions_{name}.csv"
            ),
            index=False,
        )

        del model

    output_json = os.path.join(
        OUTPUT_DIR, "patient_level_comparison.json"
    )
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved comparison:", output_json)
    print("Patient prediction CSV files saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
