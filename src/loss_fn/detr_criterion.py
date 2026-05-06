import torch
import torch.nn as nn
from torchvision.ops import giou_loss

class DETRCriterion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls_weight = config.cls_weight
        self.l1_weight = config.l1_weight
        self.iou_weight = config.iou_weight

        self.bce = torch.nn.BCELoss()
        self.l1 = torch.nn.L1Loss()
        self.iou = giou_loss()

    def forward(self, x):
        return x