import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.models.losses.utils import weight_reduce_loss
from mmengine.structures import InstanceData
from torch import Tensor
from mmdet.structures.bbox import bbox_overlaps, bbox_xyxy_to_cxcywh
from typing import Optional, Union
from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost


@MODELS.register_module()
class ContrastiveLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 margin=0.75,
                 loss_weight=1.0):

        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction
        self.margin = margin
        self.loss_weight = loss_weight


    def forward(self,
                cls_scores, 
                mask_preds,
                batch_gt_instances,
                batch_img_metas,
                weight=None,
                reduction_override=None,
                avg_factor=None):
        

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        def get_ood_mask(gt_instances):
            if (gt_instances.labels == 19).any():
                return gt_instances.masks[gt_instances.labels == 19]
            else:
                return gt_instances.masks.new_zeros((1, *gt_instances.masks.shape[-2:]))
        
        ood_mask = torch.stack([get_ood_mask(gt_instances) for gt_instances in batch_gt_instances]).squeeze(1)
                        
        mask_preds = F.interpolate(
            mask_preds, size=ood_mask.shape[-2:], mode='bilinear', align_corners=False)
        cls_scores = F.softmax(cls_scores, dim=-1)[..., :-1]
        mask_preds = mask_preds.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_scores, mask_preds)
        
        score = -torch.max(seg_logits[:, :19], dim=1)[0]  
        ood_score = score[ood_mask == 1]
        id_score = score[ood_mask == 0]
        assert ((ood_mask == 0) | (ood_mask == 1)).all()
        loss = torch.pow(id_score, 2).mean()
        if ood_mask.sum() > 0:
            loss = loss + torch.pow(torch.clamp(self.margin - ood_score, min=0.0), 2).mean()

        loss = self.loss_weight * loss

        return loss


@MODELS.register_module()
class RbALoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):

        super(RbALoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                cls_scores, 
                mask_preds,
                batch_gt_instances,
                batch_img_metas,
                weight=None,
                reduction_override=None,
                avg_factor=None):
        
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        def get_ood_mask(gt_instances):
            if (gt_instances.labels >= 19).any():
                return gt_instances.masks[gt_instances.labels >= 19].sum(dim=0)
            else:
                return gt_instances.masks.new_zeros(*gt_instances.masks.shape[-2:])

        ood_mask = torch.stack([get_ood_mask(gt_instances) for gt_instances in batch_gt_instances])
                
        mask_preds = F.interpolate(
            mask_preds, size=ood_mask.shape[-2:], mode='bilinear', align_corners=False)
        cls_scores = F.softmax(cls_scores, dim=-1)[..., :-1]
        mask_preds = mask_preds.sigmoid()
        seg_logits = torch.einsum('bqc, bqhw->bchw', cls_scores, mask_preds)
        score = seg_logits.tanh()
        score = -score.sum(dim=1)
        
        ood_score = score[ood_mask == 1]
        id_score = score[ood_mask == 0]
        
        inlier_upper_threshold = 0
        outlier_lower_threshold = 5
        
        loss = torch.pow(F.relu(id_score - inlier_upper_threshold), 2).mean()
        if ood_mask.sum() > 0:
            loss = loss + torch.pow(F.relu(outlier_lower_threshold - ood_score), 2).mean()
        
        loss = self.loss_weight * loss

        return loss


@TASK_UTILS.register_module()
class VanishingPointGuidedCost(BaseMatchCost):

    def __init__(self, vanishing_point = [512, 1024], weight: Union[float, int] = 1.):
        super().__init__(weight=weight)
        self.vp = vanishing_point
    
    def calc_angle(self, p1, p2):
        """计算两点与消失点构成的直线夹角"""
        v1 = np.array([p1[0] - self.vp[0], p1[1] - self.vp[1]])
        v2 = np.array([p2[0] - self.vp[0], p2[1] - self.vp[1]])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.arccos(cos_angle)

    def __call__(self,
                 prev_pred_instances: InstanceData,
                 curr_pred_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs):
        prev_pred_bboxes = prev_pred_instances.bboxes
        curr_pred_bboxes = curr_pred_instances.bboxes

        # avoid fp16 overflow
        if pred_bboxes.dtype == torch.float16:
            fp16 = True
            pred_bboxes = pred_bboxes.to(torch.float32)
        else:
            fp16 = False

        overlaps = bbox_overlaps(
            pred_bboxes, gt_bboxes, mode=self.iou_mode, is_aligned=False)

        if fp16:
            overlaps = overlaps.to(torch.float16)

        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight
