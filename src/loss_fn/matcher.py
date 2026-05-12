import torch.nn as nn
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou, box_convert

class HungarianMatcher(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cost_class = config.matcher.cost_class
        self.cost_bbox = config.matcher.cost_bbox
        self.cost_giou = config.matcher.cost_giou
    
    @torch.no_grad()
    def forward(self, outs, targets):
        B, N, C = outs['pred_logits'].shape
        out_prob = outs['pred_logits'].flatten(0, 1).softmax(-1)
        out_bbox = outs['pred_boxes'].flatten(0, 1)

        tgt_bbox = torch.cat([target['boxes'] for target in targets])
        tgt_ids = torch.cat([target['labels'] for target in targets])
        
        ## Classification Cost
        cost_class = -out_prob[:, tgt_ids]
        
        ## L1 Cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        ## GIoU Cost
        cost_giou = -generalized_box_iou(box_convert(boxes=out_bbox, in_fmt='cxcywh', out_fmt='xyxy'),
                                         box_convert(boxes=tgt_bbox, in_fmt='cxcywh', out_fmt='xyxy'))
        
        cost_matrix = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * self.cost_giou
        cost_matrix = cost_matrix.view(B, N, -1).cpu()
        num = [len(target['boxes']) for target in targets]
        indices = [linear_sum_assignment(c[idx]) for idx, c in enumerate(cost_matrix.split(num, -1))]
        
        return [(torch.as_tensor(i), torch.as_tensor(j)) for i,j in indices]