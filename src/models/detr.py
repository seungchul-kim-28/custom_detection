from ..layers.detr_decoder import DETRDecoder
from ..layers.detr_encoder import DETREncoder
from .resnet import ResNet
import torch.nn as nn

class DETR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = ResNet(config)
        self.encoder = DETREncoder(config)
        self.decoder = DETRDecoder(config)

    def forward(self, x):
        b_out = self.backbone(x)
        e_out = self.encoder(b_out)
        d_out = self.decoder(e_out)
        return d_out