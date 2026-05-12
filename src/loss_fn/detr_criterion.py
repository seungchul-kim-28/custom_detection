import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import giou_loss

class DETRCriterion(nn.Module):
    def __init__(self, config, matcher):
        super().__init__()
        self.cls_weight = config.criterion.cls_weight
        self.box_weight = config.criterion.box_weight
        self.iou_weight = config.criterion.iou_weight
        self.losses = config.criterion.loss_name
        self.matcher = matcher
        

    def forward(self, outs, targets):
        outs_without_aux = {name: tensor for name, tensor in outs.items() if name != 'aux_outputs'}
        indices = self.matcher(outs_without_aux, targets) 
        import ipdb; ipdb.set_trace()
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outs, targets, indices))
        
        if 'aux_outputs' in outs:
            for aux_output in outs['aux_outputs']:
                indices = self.matcher(aux_output, targets)
                for loss in self.losses:
                    dd = self.get_loss(loss, aux_output, targets, indices)
        return None
    
    def get_loss(self, name, outs, tgts, indices):
        loss = {
            'labels': self.loss_labels,
            'boxes' : self.loss_boxes
        }
        return loss[name](outs, tgts, indices)
    
    def loss_labels(self, outs, tgts, indices):
        import ipdb; ipdb.set_trace()
        return 0
    
    def loss_boxes(self, outs, tgts, indices):
        import ipdb; ipdb.set_trace()
        return 0