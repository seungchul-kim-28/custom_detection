from src.layers.detr_decoder import DETRDecoder
from src.layers.detr_encoder import DETREncoder
from src.models.resnet import ResNet
from src.models.positional_encoding import PositionalEncoding
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
        self.inp_proj = nn.Conv2d(config.model.embed_dims, config.model.embed_dims, kernel_size=1)
        self.positional_encoding = PositionalEncoding(num_pos_feats=config.model.num_pos_feat)
        self.query_embed = nn.Embedding(num_embeddings=config.model.num_queries,
                                        embedding_dim=config.model.embed_dims).weight
        
    def forward(self, x):
        b_out = self.backbone(x)
        B, D, H, W = b_out.shape
        pos = self.positional_encoding(b_out).flatten(2).permute(2,0,1)
        query_embed = self.query_embed.unsqueeze(0).repeat(B, 1, 1).permute(1,0,2)
        src = self.inp_proj(b_out).flatten(2).permute(2,0,1)
        tgt = torch.zeros_like(query_embed)

        e_out = self.encoder(src, pos)

        out = self.decoder(query=tgt,
                             memory=e_out,
                             query_pos=query_embed,
                             pos=pos)
        import ipdb; ipdb.set_trace()
        return out