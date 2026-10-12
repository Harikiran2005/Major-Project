import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# CNN FEATURE EXTRACTOR
# ============================================================

class CNNFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        # Load pretrained ResNet18
        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)

        # Remove the final classification layer
        # Output will be 512-dimensional features
        self.cnn = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        # Freeze CNN
        for param in self.cnn.parameters():
            param.requires_grad = False

    def forward(self, x):

        # Input:
        # [B, 3, 224, 224]
        #
        # B = batch size
        # 3 = three adjacent CT slices

        B, S, H, W = x.shape

        # Process each CT slice independently
        x = x.reshape(B * S, 1, H, W)

        # ResNet18 expects 3-channel input.
        # Convert grayscale CT slice to 3 channels.
        x = x.repeat(1, 3, 1, 1)

        # Extract CNN features
        with torch.no_grad():
            features = self.cnn(x)

        # Current shape:
        # [B*S, 512, 1, 1]

        features = features.flatten(1)

        # Reshape back into slices
        # [B, 3, 512]

        features = features.reshape(
            B,
            S,
            512
        )

        return features


# ============================================================
# SLICE TRANSFORMER
# ============================================================

class SliceTransformer(nn.Module):

    def __init__(
        self,
        feature_dim=512,
        transformer_dim=128,
        num_heads=4,
        num_layers=2,
        num_classes=4
    ):
        super().__init__()

        # Reduce CNN feature dimension
        self.projection = nn.Linear(
            feature_dim,
            transformer_dim
        )

        # Positional embedding
        # One position for each of the 3 CT slices
        self.positional_embedding = nn.Parameter(
            torch.randn(
                1,
                3,
                transformer_dim
            )
        )

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dropout=0.2,
            batch_first=True
        )

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Four disease classes:
        # 0 = Normal
        # 1 = COVID
        # 2 = CAP
        # 3 = Lung Cancer

        self.classifier = nn.Linear(
            transformer_dim,
            num_classes
        )

    def forward(self, features):

        # Input:
        # [B, 3, 512]

        # Project to Transformer dimension
        x = self.projection(features)

        # Add positional information
        x = x + self.positional_embedding

        # Transformer
        x = self.transformer(x)

        # Mean pooling across the 3 slices
        x = x.mean(dim=1)

        # Classification
        output = self.classifier(x)

        return output


# ============================================================
# COMPLETE CNN + TRANSFORMER MODEL
# ============================================================

class CNNTransformer(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        self.cnn = CNNFeatureExtractor()

        self.transformer = SliceTransformer(
            feature_dim=512,
            transformer_dim=128,
            num_heads=4,
            num_layers=2,
            num_classes=num_classes
        )

    def forward(self, x):

        # CNN
        features = self.cnn(x)

        # Transformer
        output = self.transformer(features)

        return output


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("2.5D CNN-TRANSFORMER MODEL TEST")
    print("=" * 60)

    # Fake batch
    x = torch.randn(
        2,
        3,
        224,
        224
    )

    print("\nInput shape:", x.shape)

    # Test CNN
    cnn = CNNFeatureExtractor()

    with torch.no_grad():
        features = cnn(x)

    print(
        "CNN feature shape:",
        features.shape
    )

    # Test Transformer
    transformer = SliceTransformer()

    with torch.no_grad():
        output = transformer(features)

    print(
        "Output shape:",
        output.shape
    )

    print(
        "\nExpected CNN feature shape: "
        "[2, 3, 512]"
    )

    print(
        "Expected output: [2, 4]"
    )

    print(
        "\n✓ Proper 2.5D CNN-Transformer "
        "forward pass successful"
    )