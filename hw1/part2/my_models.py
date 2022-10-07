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


# ref: https://blog.csdn.net/gbz3300255/article/details/105582572
class VGG16_FCN8(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        feats = list(vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features.children())
        self.feat1 = nn.Sequential(*feats[0:4])
        self.feat2 = nn.Sequential(*feats[5:9])
        self.feat3 = nn.Sequential(*feats[10:16])
        self.feat4 = nn.Sequential(*feats[17:23])
        self.feat5 = nn.Sequential(*feats[24:30])

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(in_channels=4096, out_channels=num_classes, kernel_size=1)
        )
        self.score_feat3 = nn.Conv2d(256, num_classes, 1)
        self.score_feat4 = nn.Conv2d(512, num_classes, 1)
        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)
        self.upscore4 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)

    def forward(self, x):
        x1 = self.feat1(x)      # 1/2
        x2 = self.feat2(x1)     # 1/4
        x3 = self.feat3(x2)     # 1/8
        x4 = self.feat4(x3)     # 1/16
        x5 = self.feat5(x4)     # 1/32
        score = self.classifier(x5)
        score_x3 = self.score_feat3(x3)
        score_x4 = self.score_feat3(x4)
        upscore2 = self.upscore2(score)
        upscore4 = self.upscore4(upscore2 + score_x4)
        upscore8 = self.upscore8(upscore4 + score_x3)
        return upscore8
