
# src/generate_gradcam.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finetune_densenet121_transformer import DenseNet121Transformer

DEVICE = torch.device("cpu")

CHECKPOINT_PATH = (
    PROJECT_ROOT / "outputs" / "best_densenet121_transformer.pth"
)

TEST_CSV = PROJECT_ROOT / "data" / "splits" / "test_final.csv"

COVID_DATA_DIR = PROJECT_ROOT / "data" / "covid_25d_v2"
CANCER_DATA_DIR = PROJECT_ROOT / "data" / "cancer_25d"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "gradcam"

CLASS_NAMES = {
    0: "Normal",
    1: "COVID",
    2: "CAP",
    3: "Lung Cancer",
}

CLASS_FOLDERS = {
    0: "normal",
    1: "covid",
    2: "cap",
}

IMAGES_PER_CLASS = 2
IMAGE_SIZE = 224


# --------------------------------------------------
# 2. GRAD-CAM IMPLEMENTATION
# --------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        self.hook = target_layer.register_forward_hook(
            self._forward_hook
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        self.gradients = None

        # Capture gradients directly from the layer output.
        # This avoids the full-backward-hook warning.
        if output.requires_grad:
            output.register_hook(self._gradient_hook)

    def _gradient_hook(self, grad):
        self.gradients = grad

    def generate(self, x, target_class=None):
        self.model.zero_grad(set_to_none=True)

        self.activations = None
        self.gradients = None

        # x shape: [1, 3, 224, 224]
        logits = self.model(x)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[:, target_class].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Grad-CAM activations or gradients were not captured."
            )

        # DenseNet receives the three CT slices as a batch of 3.
        # Select the middle slice, index 1.
        activations = self.activations[1:2]
        gradients = self.gradients[1:2]

        # Average gradients over spatial dimensions.
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # Weighted combination of feature maps.
        cam = (weights * activations).sum(
            dim=1,
            keepdim=True,
        )

        cam = F.relu(cam)

        # Resize heatmap to original image dimensions.
        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]

        # Normalize heatmap.
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy(), logits.detach()

    def close(self):
        self.hook.remove()


# --------------------------------------------------
# 3. CHECKPOINT LOADING
# --------------------------------------------------

def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = DenseNet121Transformer().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    # Support either a direct state_dict or a wrapped checkpoint.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    # Support checkpoints saved using DataParallel.
    if checkpoint and all(
        key.startswith("module.") for key in checkpoint
    ):
        checkpoint = {
            key.removeprefix("module."): value
            for key, value in checkpoint.items()
        }

    model.load_state_dict(checkpoint)
    model.eval()

    print("Model checkpoint loaded successfully.")
    print(f"Device: {DEVICE}")

    return model


# --------------------------------------------------
# 4. FIND CT WINDOWS FOR A PATIENT
# --------------------------------------------------

def resolve_patient_files(patient_id, label):
    patient_id = str(patient_id).strip()

    if label == 3:
        # Cancer preprocessing uses patient_id_25d.npy.
        exact_path = CANCER_DATA_DIR / f"{patient_id}_25d.npy"

        if exact_path.exists():
            return [exact_path]

        # Fallback for differently suffixed cancer files.
        if CANCER_DATA_DIR.exists():
            return sorted(
                path
                for path in CANCER_DATA_DIR.glob(
                    f"{patient_id}_*.npy"
                )
                if path.is_file()
            )

        return []

    folder_name = CLASS_FOLDERS.get(label)

    if folder_name is None:
        return []

    class_dir = COVID_DATA_DIR / folder_name

    if not class_dir.exists():
        print(f"Missing directory: {class_dir}")
        return []

    # Use a delimiter to avoid matching IDs such as P12 with P123.
    return sorted(
        path
        for path in class_dir.glob(f"{patient_id}_*.npy")
        if path.is_file()
    )


# --------------------------------------------------
# 5. GENERATE AND SAVE VISUALIZATION
# --------------------------------------------------

def save_gradcam(
    patient_id,
    array,
    label,
    heatmap,
    logits,
):
    # Input array must have shape (3, 224, 224).
    if array.shape != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Unexpected CT array shape for {patient_id}: "
            f"{array.shape}"
        )

    # Correctly select the middle CT slice.
    slice_image = array[1]

    predicted_class = int(logits.argmax(dim=1).item())
    probabilities = torch.softmax(logits, dim=1)[0]
    confidence = float(probabilities[predicted_class].item())

    # Create a safe filename.
    safe_id = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in str(patient_id)
    )

    output_path = OUTPUT_DIR / (
        f"{CLASS_NAMES[label].replace(' ', '_')}_{safe_id}_gradcam.png"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(slice_image, cmap="gray")
    axes[0].set_title(
        f"Middle CT Slice\nTrue: {CLASS_NAMES[label]}"
    )
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(slice_image, cmap="gray")
    axes[2].imshow(
        heatmap,
        cmap="jet",
        alpha=0.4,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(
        f"Overlay\nPredicted: {CLASS_NAMES[predicted_class]}\n"
        f"Confidence: {confidence:.2%}"
    )
    axes[2].axis("off")

    fig.suptitle(
        f"Patient: {patient_id}",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(
        f"  True class: {CLASS_NAMES[label]} | "
        f"Predicted: {CLASS_NAMES[predicted_class]} | "
        f"Confidence: {confidence:.2%}"
    )


# --------------------------------------------------
# 6. MAIN EXECUTION
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found: {TEST_CSV}"
        )

    if not COVID_DATA_DIR.exists():
        raise FileNotFoundError(
            f"COVID/normal/CAP data directory not found: "
            f"{COVID_DATA_DIR}"
        )

    test_df = pd.read_csv(TEST_CSV)

    required_columns = {"patient_id", "label"}
    missing_columns = required_columns - set(test_df.columns)

    if missing_columns:
        raise ValueError(
            f"Test CSV is missing columns: {missing_columns}"
        )

    model = load_model()

    # Final DenseNet convolutional block.
    target_layer = model.cnn.features.denseblock4

    gradcam = GradCAM(model, target_layer)

    generated_per_class = {
        label: 0 for label in CLASS_NAMES
    }

    try:
        for label, class_name in CLASS_NAMES.items():
            print(f"\nProcessing class: {class_name}")

            class_rows = test_df[
                pd.to_numeric(
                    test_df["label"],
                    errors="coerce",
                ) == label
            ].drop_duplicates(subset=["patient_id"])

            if class_rows.empty:
                print("No patients listed for this class.")
                continue

            for _, row in class_rows.iterrows():
                if generated_per_class[label] >= IMAGES_PER_CLASS:
                    break

                patient_id = str(row["patient_id"]).strip()

                files = resolve_patient_files(patient_id, label)

                if not files:
                    print(
                        f"Skipping {patient_id}: "
                        "no matching preprocessed files."
                    )
                    continue

                # Use the first available window for this patient.
                array_path = files[0]

                try:
                    array = np.load(array_path)
                except Exception as exc:
                    print(
                        f"Skipping {patient_id}: "
                        f"could not load array: {exc}"
                    )
                    continue

                if array.shape != (3, IMAGE_SIZE, IMAGE_SIZE):
                    print(
                        f"Skipping {patient_id}: "
                        f"unexpected shape {array.shape}"
                    )
                    continue

                # Model input shape: [batch, slices, height, width].
                x = torch.from_numpy(
                    np.asarray(array, dtype=np.float32)
                ).unsqueeze(0).to(DEVICE)

                try:
                    heatmap, logits = gradcam.generate(x)

                    save_gradcam(
                        patient_id=patient_id,
                        array=array,
                        label=label,
                        heatmap=heatmap,
                        logits=logits,
                    )

                    generated_per_class[label] += 1

                except Exception as exc:
                    print(
                        f"Grad-CAM failed for {patient_id}: {exc}"
                    )

    finally:
        gradcam.close()

    print("\n" + "=" * 55)
    print("GRAD-CAM GENERATION SUMMARY")
    print("=" * 55)

    for label, class_name in CLASS_NAMES.items():
        print(
            f"{class_name}: {generated_per_class[label]} "
            f"visualization(s)"
        )

    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
