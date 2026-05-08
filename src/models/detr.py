from ..layers.detr_decoder import DETRDecoder
from ..layers.detr_encoder import DETREncoder
from .resnet import ResNet
import torch.nn as nn
import math

class DETR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = ResNet(config)
        self.encoder = DETREncoder(config)
        self.decoder = DETRDecoder(config)
        # import ipdb; ipdb.set_trace()
        self.inp_proj = nn.Conv2d(config.model.dim_ffn, config.model.embed_dims, kernel_size=1)

    def forward(self, x):
        b_out = self.backbone(x)
        import ipdb; ipdb.set_trace()
        e_out = self.encoder(self.inp_proj(b_out))
        d_out = self.decoder(e_out)
        return d_out
    
class PositionalEncoding(nn.Module):
    def __init__(self, temperature, normalize=True,):
        super().__init__()
        self.temp = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x):
        return None