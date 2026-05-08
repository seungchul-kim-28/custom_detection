import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import giou_loss

class DETRCriterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls_weight = config.criterion.cls_weight
        self.box_weight = config.criterion.box_weight
        self.iou_weight = config.criterion.iou_weight


    def forward(self, x):
        return x