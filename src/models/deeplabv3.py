
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabV3(nn.Module):
    def __init__(self, pretrained_backbone=False):
        super().__init__()

        # Khởi tạo ResNet-50 backbone.
        # False giúp chạy thử mà không cần tải weights từ Internet.
        self.model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
            num_classes=21,
            aux_loss=None,
        )

        # Chuyển classifier từ 21 lớp sang 1 lớp:
        # foreground = lưỡi, background = nền.
        classifier = self.model.classifier
        in_channels = classifier[-1].in_channels
        classifier[-1] = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1,
        )

        if pretrained_backbone:
            raise NotImplementedError(
                "Pretrained backbone chưa được bật trong phiên bản này."
            )

    def forward(self, x):
        return self.model(x)["out"]
