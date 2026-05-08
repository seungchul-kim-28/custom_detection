import torch.nn as nn
import torch
from src.models.detr import with_pos_embed, PositionalEncoding

class DETREncoderLayer(nn.Module):
    def __init__(self, config, with_pos_embed):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=config.model.embed_dims,
                                               num_heads=config.model.num_heads,)
        self.cross_attn = nn.MultiheadAttention(embed_dim=config.model.embed_dims,
                                                num_heads=config.model.num_heads)
        # self.embed_dims = embed_dims
        self.norm1 = nn.LayerNorm(config.model.embed_dims)
        self.drop1 = nn.Dropout(config.model.drop_rate)
        self.norm2 = nn.LayerNorm(config.model.embed_dims)
        self.drop2 = nn.Dropout(config.model.drop_rate)
        self.ffn = nn.Linear(config.model.embed_dims, config.model.embed_dims)

    def forward(self, query, key, value):
        sa_out = self.self_attn(query, key, value)
        query = query + sa_out
        norm1 = self.norm1(query)
        ffn = self.ffn(norm1)
        ffn = ffn + norm1
        return ffn

class DETREncoder(nn.Module):
    def __init__(self, config,):
        super().__init__()
        self.use_aux = config.model.use_aux
        self.layers = nn.ModuleList(
            DETREncoderLayer(config) for _ in range(int(config.model.num_enc_layers))
        )
        
    
    def forward(self, x, pos):
        out = x
        if self.use_aux:
            res = []
        for layer in self.layers:
            out = layer(query=with_pos_embed(out, pos), 
                        key=with_pos_embed(out, pos),
                        value = out)
            if self.use_aux:
                res.append(out)

        return res if self.use_aux else out
 