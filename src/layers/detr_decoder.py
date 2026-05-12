import torch.nn as nn
import torch
# from src.models.detr import with_pos_embed, PositionalEncoding
# from src.models.positional_encoding import with_pos_embed


class DETRDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_aux = config.model.use_aux
        self.self_attn = nn.MultiheadAttention(embed_dim=config.model.embed_dims,
                                               num_heads=config.model.num_heads,)
        self.cross_attn = nn.MultiheadAttention(embed_dim=config.model.embed_dims,
                                                num_heads=config.model.num_heads)
        # self.embed_dims = embed_dims
        self.norm1 = nn.LayerNorm(config.model.embed_dims)
        self.drop1 = nn.Dropout(config.model.drop_rate)
        self.norm2 = nn.LayerNorm(config.model.embed_dims)
        self.drop2 = nn.Dropout(config.model.drop_rate)
        self.norm3 = nn.LayerNorm(config.model.embed_dims)
        self.drop3 = nn.Dropout(config.model.drop_rate)
        self.ffn = nn.Linear(config.model.embed_dims, config.model.embed_dims)
    
    @staticmethod
    def with_pos_embed(query, pos):
        return query if pos is None else query + pos
    
    def forward(self, query, memory, pos, query_pos):
        # import ipdb; ipdb.set_trace()
        sa1 = self.self_attn(query = self.with_pos_embed(query, query_pos),
                             key = self.with_pos_embed(query, query_pos),
                             value = query)[0]
        norm1 = self.norm1(query + self.drop1(sa1))

        ca1 = self.cross_attn(query = self.with_pos_embed(norm1, query_pos), 
                              key = self.with_pos_embed(memory, pos),
                              value = memory)[0]
        norm2 = self.norm2(norm1 + self.drop2(ca1))
        ffn = self.ffn(norm2)
        norm3 = self.norm3(self.drop3(ffn) + norm2)
        return norm3

class DETRDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_aux = config.model.use_aux

        self.layers = nn.ModuleList(
            DETRDecoderLayer(config) for i in range(int(config.model.num_dec_layers))
            )
        
        self.cls_mlp = nn.Linear(config.model.embed_dims, config.base.dataset.num_classes)
        self.box_mlp = nn.Sequential(
            nn.Linear(config.model.embed_dims, config.model.embed_dims),
            nn.Linear(config.model.embed_dims, 4)
        )
    
    def forward(self, query, query_pos, memory, pos):
        res = []
        out = query
        for idx, layer in enumerate(self.layers):
            out = layer(query = out, 
                        memory = memory, 
                        pos = pos,
                        query_pos = query_pos)
        
            if self.use_aux:
                res.append(out.permute(1,0,2))

        if self.use_aux:
            output_logits = self.cls_mlp(torch.stack(res))
            output_boxes = self.box_mlp(torch.stack(res)).sigmoid()
            outputs = {'pred_logits': output_logits[-1], 'pred_boxes': output_boxes[-1]}
            outputs['aux_outputs'] = self.set_aux_loss(output_logits[:-1], output_boxes[:-1])
            return outputs
        
        output_logits = self.cls_mlp(out)
        output_boxes = self.box_mlp(out)
        
        return {'pred_logits': output_logits, 'pred_boxes': output_boxes}
    
    @staticmethod
    def set_aux_loss(output_logits, output_boxes):
        return [{'pred_logits': logits, 'pred_boxes': boxes}
                for logits, boxes in zip(output_logits, output_boxes)]
 