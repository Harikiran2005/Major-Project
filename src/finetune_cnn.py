
class FineTuneCNN(nn.Module):

    def __init__(self):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT
        resnet = resnet18(weights=weights)

        # Freeze all CNN layers initially
        for param in resnet.parameters():
            param.requires_grad = False

        # Fine-tune the final CNN block
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        # Remove the ImageNet classification head
        self.cnn = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        # Standard ImageNet normalization
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406])
            .view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225])
            .view(1, 3, 1, 1)
        )

    def forward(self, x):
        # Input: [B, 1, 224, 224]

        # Convert grayscale CT to three channels
        x = x.repeat(1, 3, 1, 1)

        # Normalize for pretrained ResNet18
        x = (x - self.mean) / self.std

        # Extract CNN features
        features = self.cnn(x)

        # Output: [B, 512]
        features = features.flatten(1)

        return features
