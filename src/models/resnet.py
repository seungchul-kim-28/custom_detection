import torch.nn as nn
from torchvision.models import ResNet101_Weights, ResNet18_Weights, ResNet50_Weights
from torchvision.models import resnet18, resnet50, resnet101


class ResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        backbone_num = int(config.model.backbone.num)
        self.body = self._build_backbone(backbone_num)
        self.out_channels = 512 if backbone_num == 18 else 2048

    @staticmethod
    def _build_backbone(backbone_num: int) -> nn.Sequential:
        if backbone_num == 18:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif backbone_num == 50:
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif backbone_num == 101:
            model = resnet101(weights=ResNet101_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported ResNet backbone: {backbone_num}")

        return nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        return self.body(x)
