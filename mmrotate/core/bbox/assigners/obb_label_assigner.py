# Copyright (c) OpenMMLab. All rights reserved.

import cv2
import torch
import torch.nn as nn

from ..builder import ROTATED_BBOX_ASSIGNERS, build_bbox_coder
from ..iou_calculators import build_iou_calculator
from mmdet.core import multi_apply

PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0), (147, 116, 116), (0, 0, 255)]
CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter')
@ROTATED_BBOX_ASSIGNERS.register_module()
class OBBLabelAssigner(nn.Module):
    def __init__(self,
                 num_classes: int,
                 topk: int = 15,
                 alpha: float = 1.0,
                 beta: float = 6.0,
                 gamma: float = 1e-7,
                 angle_version='le90',
                 featmap_strides = None,
                 regress_ranges = None,
                 bbox_coder = dict(type='DistanceAnglePointCoder', angle_version='le90'),
                 iou_calculator=dict(type='RBboxOverlaps2D')):
        super().__init__()
        
        ''' only 9-1 (9-1 with VFL is 9-2) '''
        
        self.num_classes = num_classes
        self.regress_ranges = regress_ranges
        self.featmap_strides = featmap_strides
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.angle_version = angle_version
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.bbox_coder = build_bbox_coder(bbox_coder)

    def _get_target_single(self, gt_bboxes, gt_labels, bbox_preds, probs, points, regress_ranges,
                           num_points_per_lvl, debug = False, img_meta = None):
        """Compute regression, classification and angle targets for a single
        image."""

        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        annotation_bbox = gt_bboxes
        bboxes = bbox_preds.clone().detach() ##
        centerness_targets = []
        # alpha = 0.5
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1))
        
        areas = gt_bboxes[:, 2] * gt_bboxes[:, 3] # wh
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
        
        delta_x = 2*offset_x/w
        delta_y = 2*offset_y/h
        centerness_targets = 1 - torch.sqrt(((delta_x**2 + delta_y**2)+ 1e-8)/2)
        centerness_targets = centerness_targets.clamp_min(0)
        
        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        radius = 1.5
        stride = offset.new_zeros(offset.shape)
        
        # condition3: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))
        
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.featmap_strides[lvl_idx] * radius
            
            ''' bbox pred scale'''
            stride_bboxes = bboxes[lvl_begin:lvl_end, :4] * self.featmap_strides[lvl_idx]
            bboxes[lvl_begin:lvl_end, :4] = stride_bboxes
            lvl_begin = lvl_end
            
        # condition2: inside a `center bbox`
        inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
        inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                inside_gt_bbox_mask)
        
        ''' label assignment 12'''
        
        areas *= inside_gt_bbox_mask ## 1
        areas *= inside_regress_range
        
        det_probs = probs.softmax(-1)
        det_rbboxes = self.bbox_coder.decode(points[:,0], bboxes) #lrtba -> ctr wh a
        inside_topk_lvl = torch.zeros_like(inside_gt_bbox_mask).float()

        for i, gt in enumerate(annotation_bbox):
            matching_cost = 0.2 * centerness_targets[:,i] +\
                            0.2 * self.iou_calculator(det_rbboxes, gt.unsqueeze(0)).squeeze(-1) +\
                            0.6 * det_probs[:, gt_labels[i]]
            matching_cost *= inside_gt_bbox_mask[:,i]
            matching_cost *= inside_regress_range[:,i]
            
            if matching_cost.shape[0] >= self.topk:
                k = self.topk
            else:
                k = matching_cost.shape[0]
                
            val, idx = matching_cost.sort(descending=True)
            idx = idx[:k]
            val = val[:k]
            inside_topk_lvl[idx,i] = val # cost가 0인 값은 False가 되도록

        areas *= inside_topk_lvl.bool()
            
        max_area, max_area_inds = areas.max(dim=1) # 겹치는 points를 늘린 후 max값 선택
        
        labels = gt_labels[max_area_inds]
        # labels[max_area == 0] = self.num_classes
        labels[max_area == 0] = 15 # jyjyjyjyjyjy
        
        if 0 in max_area_inds.bincount().tolist(): # 0개의 anchorpoint가 할당된 경우
            noacp = torch.where(max_area_inds[max_area != 0].bincount() == 0)[0] # GT 중 match가 안된 gt ind
            rematchig_anchor = centerness_targets.argmax(dim=0)[noacp]
            max_area_inds[rematchig_anchor] = noacp # ctr기반으로 할당
            labels[rematchig_anchor] = gt_labels[noacp]
            
        bbox_targets = bbox_targets[range(num_points), max_area_inds]
        angle_targets = gt_angle[range(num_points), max_area_inds]
        
        # if debug:
        #     ig = cv2.imread(img_meta['filename'])
        #     ig = cv2.cvtColor(ig, cv2.COLOR_BGR2RGB)
        #     for i, lb in enumerate(labels):
        #         key = str(lb.item())
        #         x = points[i][0][0].detach().item()
        #         y = points[i][0][1].detach().item()
        #         if key == '15':
        #             cv2.circle(ig, (int(x), int(y)), 1, (0, 0, 0), 0, 1)
                    
        #         else:
        #             cv2.circle(ig,  (int(x), int(y)), 3, PALETTE[int(key)], -1, 1)
                
        #     for i, p in enumerate(PALETTE):
        #         cv2.putText(ig, CLASSES[i], (20, 20*i+10), 1, 1, p)

        #     cv2.imwrite(f'demo/la/'+ img_meta['ori_filename'], ig)
        
        return labels, bbox_targets, angle_targets

    @torch.no_grad()
    def forward(self, points, gt_bboxes_list, gt_labels_list, 
                    bbox_preds, angle_preds, cls_scores, img_meta):

        # assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        
        bn = bbox_preds[0].shape[0]
        rbbox_preds = [
            torch.cat([b.view(bn, 4, -1).permute(0,2,1), 
                       a.view(bn, 1, -1).permute(0,2,1)], dim=-1)
            for b, a in zip(bbox_preds, angle_preds)
        ]
        # rbbox_probs = [torch.cat([p.view(bn, 15, -1).permute(0,2,1)], dim=-1) for p in cls_scores] # jyjyjy
        rbbox_probs = [torch.cat([p.view(bn, self.num_classes, -1).permute(0,2,1)], dim=-1) for p in cls_scores]
        concat_rbboxes = [preds for preds in torch.cat(rbbox_preds, dim=1)]
        concat_probs = [probs for probs in torch.cat(rbbox_probs, dim=1)]
        
        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]
        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, angle_targets_list= multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            concat_rbboxes, 
            concat_probs,
            points=concat_points,  
            regress_ranges=concat_regress_ranges, 
            num_points_per_lvl= num_points,
            debug = False,
            img_meta = img_meta) 
        
        # split to per img, per level
        
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        angle_targets_list = [
            angle_targets.split(num_points, 0)
            for angle_targets in angle_targets_list
        ]
        
        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            bbox_targets /= self.featmap_strides[i]
            
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
            
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_angle_targets)


