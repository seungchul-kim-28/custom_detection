import torch.nn as nn
import torch
# from src.models.detr import with_pos_embed, PositionalEncoding
# from src.models.positional_encoding import with_pos_embed

class DETREncoderLayer(nn.Module):
    def __init__(self, config,):
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
        # import ipdb; ipdb.set_trace()
        sa_out = self.self_attn(query, key, value)[0]
        norm1 = self.norm1(query + self.drop1(sa_out))
        ffn = self.ffn(norm1)
        norm2 = self.norm2(ffn + self.drop2(norm1))
        return norm2

class DETREncoder(nn.Module):
    def __init__(self, config,):
        super().__init__()
        self.layers = nn.ModuleList(
            DETREncoderLayer(config) for _ in range(int(config.model.num_enc_layers))
        )
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, x, pos):
        out = x
        
        for layer in self.layers:
            out = layer(query=self.with_pos_embed(out, pos), 
                        key=self.with_pos_embed(out, pos),
                        value = out)
        
        return out
 