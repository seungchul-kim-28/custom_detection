from ..layers.detr_decoder import DETRDecoder
from ..layers.detr_encoder import DETREncoder
from .resnet import ResNet
import torch
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
        self.positional_encoding = PositionalEncoding()

    def forward(self, x):
        b_out = self.backbone(x)
        import ipdb; ipdb.set_trace()
        pos = self.positional_encoding(b_out)
        e_out = self.encoder(self.inp_proj(b_out), pos)
        d_out = self.decoder(e_out, pos)
        return d_out
    
class PositionalEncoding(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True,):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temp = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x):
        B, _, H, W = x.shape
        mask = torch.zeros((B, H, W), dtype=torch.bool, device=x.device)

        y_embed = ~mask.cumsum(dim=1, dtype=torch.float32)
        x_embed = ~mask.cumsum(dim=2, dtype=torch.float32)

        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=x.device,
        )

        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
                            dim=4).flatten(3)

        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
                            dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3)

        pos = pos.permute(0, 3, 1, 2)

        return pos
    
def with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor+pos