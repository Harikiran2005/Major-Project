import torch
import torch.nn as nn
from torchvision import models


class CNNFeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Remove final classification layer
        self.cnn = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        # Freeze CNN
        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, x):

        # x:
        # [B, 3, 224, 224]

        batch_size = x.size(0)

        # Treat each of the 3 CT slices separately
        slices = x.reshape(
            batch_size * 3,
            1,
            x.size(2),
            x.size(3)
        )

        # ResNet expects 3 channels.
        # Replicate grayscale CT slice.

        slices = slices.repeat(
            1,
            3,
            1,
            1
        )

        with torch.no_grad():

            features = self.cnn(
                slices
            )

        # [B*3, 512, 1, 1]

        features = features.flatten(
            1
        )

        # [B*3, 512]

        features = features.reshape(
            batch_size,
            3,
            512
        )

        # [B, 3, 512]

        return features


class SliceTransformer(nn.Module):

    def __init__(
        self,
        num_classes=4,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.2
    ):

        super().__init__()

        # ====================================================
        # PROJECT CNN FEATURES
        # ====================================================

        self.feature_projection = nn.Linear(
            512,
            embed_dim
        )

        # ====================================================
        # POSITIONAL EMBEDDING
        # ====================================================

        self.positional_embedding = nn.Parameter(
            torch.randn(
                1,
                3,
                embed_dim
            )
        )

        # ====================================================
        # TRANSFORMER
        # ====================================================

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ====================================================
        # CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(
                embed_dim,
                num_classes
            )
        )

    def forward(self, features):

        # features:
        # [B, 3, 512]

        features = self.feature_projection(
            features
        )

        # [B, 3, 128]

        features = (
            features +
            self.positional_embedding
        )

        # ====================================================
        # TRANSFORMER
        # ====================================================

        features = self.transformer(
            features
        )

        # [B, 3, 128]

        # ====================================================
        # AGGREGATE SLICES
        # ====================================================

        features = features.mean(
            dim=1
        )

        # [B, 128]

        # ====================================================
        # CLASSIFICATION
        # ====================================================

        output = self.classifier(
            features
        )

        # [B, 4]

        return output


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CNN + TRANSFORMER TEST")
    print("=" * 60)

    # CNN
    cnn = CNNFeatureExtractor()

    # Transformer
    transformer = SliceTransformer(
        num_classes=4
    )

    # Dummy input

    x = torch.randn(
        2,
        3,
        224,
        224
    )

    print(
        "\nInput:",
        x.shape
    )

    # Extract features

    features = cnn(x)

    print(
        "CNN features:",
        features.shape
    )

    # Transformer

    with torch.no_grad():

        output = transformer(
            features
        )

    print(
        "Output:",
        output.shape
    )

    print(
        "\nExpected CNN features:",
        "[2, 3, 512]"
    )

    print(
        "Expected output:",
        "[2, 4]"
    )

    print(
        "\n✓ CNN + Transformer test successful"
    )