import torch.nn as nn
import torch




class DETRDecoderLayer(nn.Module):
    def __init__(self, config):
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

    def forward(self, x):
        return None

class DETRDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_aux = config.model.use_aux

        self.layers = nn.ModuleList(DETRDecoderLayer(config) for i in range(int(config.model.num_dec_layers)))
        
        if self.use_aux:
            self.cls_proj = nn.ModuleList(
                nn.Linear(config.model.embed_dims, config.base.dataset.num_classes) for _ in range(int(config.model.num_dec_layers))
                )
            
            self.box_mlp = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(config.model.embed_dims, config.model.embed_dims),
                    nn.Linear(config.model.embed_dims, 4)) for _ in range(int(config.model.num_dec_layers))
                )
        else:
            self.cls_proj = nn.Linear(config.model.embed_dims, config.base.dataset.num_classes)
            self.box_mlp = nn.Sequential(
                nn.Linear(config.model.embed_dims, config.model.embed_dims),
                nn.Linear(config.model.embed_dims, 4)
            )
    
    def forward(self, query, memory):
        pred_logits = []
        pred_boxes = []
        for idx, layer in enumerate(self.layers):
            out = layer(query)

            if self.use_aux:
                pred_logits.append(self.cls_proj[idx](out))
                pred_boxes.append(self.box_mlp[idx](out))
        
        if not self.use_aux:
            pred_logits = self.cls_proj(out)
            pred_boxes = self.box_mlp(out)


        return None
 