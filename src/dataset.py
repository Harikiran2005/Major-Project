
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LungDiseaseDataset(Dataset):
    def __init__(self, csv_file, covid_data_dir, cancer_data_dir):
        self.data = pd.read_csv(csv_file)
        self.covid_data_dir = covid_data_dir
        self.cancer_data_dir = cancer_data_dir
        self.samples = []

        for _, row in self.data.iterrows():
            patient_id = str(row["patient_id"]).strip()
            class_name = str(row["class_name"]).strip().lower()
            label = int(row["label"])

            # Prefer an explicit file path from the manifest.
            explicit_path = str(row.get("filepath", "")).strip()
            if explicit_path and explicit_path.lower() not in {"nan", "none"}:
                filepath = os.path.normpath(explicit_path)
                if os.path.isfile(filepath):
                    self.samples.append((filepath, label))
                else:
                    print(f"WARNING: missing file: {filepath}")
                continue

            # Original cancer files: one 2.5D array per patient.
            if class_name == "cancer":
                filename = f"{patient_id}_25d.npy"
                filepath = os.path.join(self.cancer_data_dir, filename)

                if os.path.isfile(filepath):
                    self.samples.append((filepath, label))
                else:
                    print(f"WARNING: cancer file not found: {filepath}")
                continue

            # Original Normal / COVID / CAP windows.
            dataset_dir = os.path.join(self.covid_data_dir, class_name)
            if not os.path.isdir(dataset_dir):
                continue

            prefix = patient_id + "_"
            for filename in os.listdir(dataset_dir):
                if filename.endswith(".npy") and filename.startswith(prefix):
                    self.samples.append(
                        (os.path.join(dataset_dir, filename), label)
                    )

        print(
            f"Loaded {len(self.samples)} samples "
            f"from {os.path.basename(csv_file)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        filepath, label = self.samples[index]

        image = np.load(filepath).astype(np.float32)

        if image.shape != (3, 224, 224):
            raise ValueError(
                f"Unexpected array shape {image.shape} in {filepath}; "
                "expected (3, 224, 224)."
            )

        if not np.isfinite(image).all():
            raise ValueError(f"NaN or infinity found in {filepath}")

        image = torch.from_numpy(image)
        label = torch.tensor(label, dtype=torch.long)

        return image, label
