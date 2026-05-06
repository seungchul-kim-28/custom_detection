import torch.nn as nn
import torch




class DETRDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dims = config.embed_dims

    def forward(self, x):
        return None

class DETRDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(embed_dim=config.embed_dims,
                                               num_heads=config.num_heads,)
        self.cross_attn = nn.MultiheadAttention(embed_dim=config.embed_dims,
                                                num_heads=config.num_heads)
        
        self.norm1 = nn.LayerNorm(config.norm_rate)
        self.drop1 = nn.Dropout(config.drop_rate)
        self.norm2 = nn.LayerNorm(config.norm_rate)
        self.drop2 = nn.Dropout(config.drop_rate)
        
        self.cls_proj = nn.Linear(config.model.embed_dims, config.base.dataset.num_classes)
        self.box_mlp = nn.Sequential(
            nn.Linear(config.model.embed_dims, config.model.embed_dims),
            nn.Linear(config.model.embed_dims, 4)
        )
    
    def forward(self, query, memory):
        q = k = v = query

        x = self.self_attn(q,k,v)

        pred_logits = self.cls_proj(x)
        pred_boxes = self.box_mlp(x)

        return None
 