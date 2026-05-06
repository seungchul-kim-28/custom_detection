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
        
    
    def forward(self, x):
        return x
 