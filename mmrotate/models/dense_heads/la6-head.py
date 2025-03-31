# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence, Union
import math

import torch

from .rotated_yolov8_head import RotatedYOLOv8Head
from ..builder import ROTATED_HEADS, build_loss
from ..blocks import *

INF = 1e8

def make_divisible(x: float,
                   widen_factor: float = 1.0,
                   divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor




@ROTATED_HEADS.register_module()
class LabelAssignment6(RotatedYOLOv8Head):
    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 reg_max = 0,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 regress_ranges=((-1, 64), (64, 128), (128, 256)), ##
                 bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='cls_preds',
                         std=0.01,
                         bias_prob=0.01)),               
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(num_classes=num_classes,
                         in_channels=in_channels,
                         widen_factor=widen_factor,
                         reg_max=reg_max,
                         featmap_strides=featmap_strides,
                         regress_ranges=regress_ranges,
                         bbox_coder=bbox_coder,
                         loss_cls=loss_cls,
                         loss_bbox=loss_bbox,
                         norm_cfg=norm_cfg, 
                         act_cfg=act_cfg,
                         init_cfg=init_cfg,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg)

    def _get_target_single(self, gt_bboxes, gt_labels, bbox_preds, probs, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression, classification and angle targets for a single
        image."""
        RIoU = build_loss(dict(type='RotatedIoULoss', loss_weight=1.0,
                               reduction='none', mode='linear')) ##
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        annotation_bbox = gt_bboxes
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1))
        
        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3] # wh
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                               dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None])
        offset = offset.squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        # condition1: inside a `center bbox`
        radius = self.center_sample_radius
        stride = offset.new_zeros(offset.shape)
        bboxes = bbox_preds.clone().detach() ##
        
        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.featmap_strides[lvl_idx] * radius
            
            ''' bbox pred scale'''
            stride_bboxes = bboxes[lvl_begin:lvl_end, :4] * self.featmap_strides[lvl_idx]
            bboxes[lvl_begin:lvl_end, :4] = stride_bboxes
            
            lvl_begin = lvl_end

        inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
        inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                inside_gt_bbox_mask)
        alpha = 0.5
        areas[inside_gt_bbox_mask == 0] = 0 
        overlap_idx = torch.where(torch.bincount(torch.where(areas != 0)[0]) >= 2)[0] # 위에서 2개이상의 bbox가 할당된 det 인덱스
        if len(overlap_idx) != 0:
            pts = points[overlap_idx]
            iou_matrix = torch.zeros(len(overlap_idx), len(annotation_bbox), device=areas.device) 
            det_overlapbboxes = self.bbox_coder.decode(pts[:,0], bboxes[overlap_idx])
            det_probs = probs[overlap_idx].softmax(-1)
            mask = (areas[overlap_idx] != 0).float()
            for i, gt in enumerate(annotation_bbox):
                cls = gt_labels[i]
                gt = gt.expand(len(overlap_idx), 5)
                iou =  1-(RIoU(det_overlapbboxes, gt)*(10**3)).round()/ (10**3)
                prob = det_probs[:, cls]
                iou_matrix[:,i] = alpha * iou + (1-alpha) * prob
            iou_matrix *= mask
            areas[overlap_idx] = iou_matrix
        max_area, max_area_inds = areas.max(dim=1) 
        labels = gt_labels[max_area_inds]
        labels[max_area == 0] = self.num_classes
        bbox_targets = bbox_targets[range(num_points), max_area_inds]
        angle_targets = gt_angle[range(num_points), max_area_inds]
           
            
        return labels, bbox_targets, angle_targets
