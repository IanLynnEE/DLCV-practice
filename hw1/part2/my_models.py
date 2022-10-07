import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class VGG16_FCN32(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.vgg16_features = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features
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


class VGG16_FCN8(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.feats = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(in_channels=4096, out_channels=num_classes, kernel_size=1)
        )
        self.upsample4 = nn.ConvTranspose2d(num_classes, 256, 8, stride=4, bias=False)
        self.upsample2 = nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False)
        self.upsample8 = nn.ConvTranspose2d(256, num_classes, 8, stride=8, bias=False)

    def forward(self, x):
        for idx, feature in enumerate(self.feats):
            x = feature(x)
            if idx == 16:
                pool3 = x
            elif idx == 23:
                pool4 = x
        score = self.classifier(x)
        # print('score', score.shape)
        # print('pool3', pool3.shape)
        # print('pool4', pool4.shape)
        upsample_score = self.upsample4(score)
        upsample_pool4 = self.upsample2(pool4)
        upsample8 = self.upsample8(pool3 + upsample_pool4 + upsample_score)
        return upsample8
