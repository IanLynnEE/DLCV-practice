import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class VGG16_FCN32(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.vgg16_features = vgg16(VGG16_Weights.IMAGENET1K_FEATURES).features
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Conv2d(in_channels=4096, out_channels=num_classes, kernel_size=1)
        )
        self.upsampling32 = nn.ConvTranspose2d(
            in_channels=num_classes,
            out_channels=num_classes,
            kernel_size=64,
            stride=32,
            bias=False
        )

    def forward(self, x):
        x = self.vgg16_features(x)
        x = self.classifier(x)
        x = self.upsampling32(x)
        return x