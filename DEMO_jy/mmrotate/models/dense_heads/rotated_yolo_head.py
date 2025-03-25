# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import random
from typing import List, Optional, Sequence, Tuple, Union
import math
import torch, gc
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
from torch import linalg
import numpy as np
from ..builder import ROTATED_HEADS, build_loss
from ..blocks import *
from mmengine.model import BaseModule
from mmengine.model import bias_init_with_prob
from mmcv.cnn import ConvModule
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
from mmcv.cnn.bricks import build_norm_layer
from mmdet.core import multi_apply, reduce_mean
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmrotate.core import build_bbox_coder, multiclass_nms_rotated, build_assigner
INF = 1e8
PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0), (147, 116, 116), (0, 0, 255)]
CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter')
def make_divisible(x: float,
                   widen_factor: float = 1.0,
                   divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor

@ROTATED_HEADS.register_module()
class RotatedYOLOv6Head(BaseDenseHead):
    """Anchor-Free Rotated Yolov6 Head predicting four side distances (without angle loss)"""
    def __init__(self,
                 num_classes : int = 15,
                 in_channels : Union[int, Sequence] = [256, 512, 1024],
                 widen_factor: float = 1.0,
                 reg_max=0,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 regress_ranges=((-1, 64), (64, 128), (128, 256)),
                 bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 matching:int=0,
                 debug:bool=False,
                #  loss_angle=dict(type='L1Loss', loss_weight=1.0),
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
                 test_cfg=None,
                 prior_match_thr: float = 4.0,
                 near_neighbor_thr: float = 0.5,
                 ignore_iof_thr: float = -1.0,
                 obj_level_weights: List[float] = [4.0, 1.0, 0.4],
                 center_sample_radius=1.5):
        super().__init__(init_cfg)

        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.reg_max = reg_max
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        # self.loss_angle = build_loss(loss_angle)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.prior_generator = MlvlPointGenerator(featmap_strides)
        # In order to keep a more general interface and be consistent with
        # anchor_head. We can think of point like one anchor
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self.regress_ranges = regress_ranges
        self.center_sample_radius = center_sample_radius
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # if isinstance(in_channels, int):
        #     self.in_channels = [int(in_channels * widen_factor)
        #                         ] * self.num_levels
        # else:
        #     self.in_channels = [int(i * widen_factor) for i in in_channels]
        self.in_channels = in_channels
        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels
        self._init_layers()

    def _init_layers(self):
        """initialize conv layers in YOLOv6 head."""
        # Init decouple head
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.ang_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.ang_preds = nn.ModuleList()
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.featmap_strides])
        self.stems = nn.ModuleList()

        if self.reg_max > 1:
            proj = torch.arange(
                self.reg_max + self.num_base_priors, dtype=torch.float)
            self.register_buffer('proj', proj, persistent=False)

        for i in range(self.num_levels):
            self.stems.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    kernel_size=1,
                    stride=1,
                    padding=1 // 2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            
            self.cls_convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    kernel_size=3,
                    stride=1,
                    padding=3 // 2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            self.reg_convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    kernel_size=3,
                    stride=1,
                    padding=3 // 2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            #rotated yolov6
            self.ang_convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.in_channels[i],
                    kernel_size=3,
                    stride=1,
                    padding=3 // 2,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=self.in_channels[i],
                    out_channels=self.num_base_priors * self.num_classes,
                    kernel_size=1))
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=self.in_channels[i],
                    out_channels=(self.num_base_priors + self.reg_max) * 4,
                    kernel_size=1))
            #rotated yolov6
            self.ang_preds.append(
                nn.Conv2d(
                    in_channels=self.in_channels[i],
                    out_channels=(self.num_base_priors + self.reg_max) * 1,
                    kernel_size=1))

    def init_weights(self):
        super().init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv in self.cls_preds:
            conv.bias.data.fill_(bias_init)
            conv.weight.data.fill_(0.)

        for conv in self.reg_preds:
            conv.bias.data.fill_(1.0)
            conv.weight.data.fill_(0.)

        for conv in self.ang_preds:
            conv.bias.data.fill_(1.0)
            conv.weight.data.fill_(0.)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions.
        """
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.stems, self.cls_convs,
                           self.cls_preds, self.reg_convs, self.reg_preds, 
                           self.ang_convs, self.ang_preds, self.scales, self.featmap_strides)

    def forward_single(self, x: Tensor, stem: nn.Module, cls_conv: nn.Module,
                       cls_pred: nn.Module, reg_conv: nn.Module,
                       reg_pred: nn.Module, ang_conv: nn.Module,
                       ang_pred: nn.Module, scale: List, stride) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        y = stem(x)
        cls_x = y
        reg_x = y
        ang_x = y
        cls_feat = cls_conv(cls_x)
        reg_feat = reg_conv(reg_x)
        ang_feat = ang_conv(ang_x)

        cls_score = cls_pred(cls_feat)
        bbox_dist_preds = reg_pred(reg_feat)
        predicted_angle = ang_pred(ang_feat)

        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max + self.num_base_priors,
                 h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = scale(bbox_dist_preds).float()
            bbox_preds = bbox_preds.clamp(min=0)
            if not self.training:
                bbox_preds *= stride
        if self.training:
            return cls_score, bbox_preds, predicted_angle
        else:
            return cls_score, bbox_preds, predicted_angle

    def special_init(self):
        """Since YOLO series algorithms will inherit from YOLOv5Head, but
        different algorithms have special initialization process.

        The special_init function is designed to deal with this situation.
        """
        if self.train_cfg:
            # Add common attributes to reduce calculation
            self.featmap_sizes_train = None
            self.num_level_priors = None
            self.flatten_priors_train = None
            self.stride_tensor = None

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            angle_preds (list[Tensor]): Box angle for each scale level, \
                each is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        gc.collect()
        torch.cuda.empty_cache()
        
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        self.img_metas = img_metas[0]

        labels, bbox_targets, angle_targets = self.get_targets(
            all_level_points, gt_bboxes, gt_labels, 
            bbox_preds, angle_preds, cls_scores,
            debug = self.debug, matching = self.matching)

        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for angle_pred in angle_preds
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
         
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])
        
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]

        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        
        # if centerness_targets:
            # flatten_centerness_targets = torch.cat(centerness_targets)
            # pos_centerness_targets = flatten_centerness_targets[pos_inds]
            
        if len(pos_inds) > 0:
            
            pos_points = flatten_points[pos_inds]
            bbox_coder = self.bbox_coder
            pos_bbox_preds = torch.cat([pos_bbox_preds, pos_angle_preds],
                                        dim=-1)
            pos_bbox_targets = torch.cat(
                [pos_bbox_targets, pos_angle_targets], dim=-1)
            pos_decoded_bbox_preds = bbox_coder.decode(pos_points,
                                                       pos_bbox_preds)
            pos_decoded_target_preds = bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds)
            
            cls_iou_targets = torch.zeros_like(flatten_cls_scores)
            if self.loss_cls.__str__() == 'VarifocalLoss()':                
                bbox_overlap = build_loss(dict(type='RotatedIoULoss', loss_weight=1.0,
                               reduction='none', mode='linear')) ##
                iou_targets = 1 - bbox_overlap(
                    pos_decoded_bbox_preds,
                    pos_decoded_target_preds.detach()).clamp(min=1e-6)
                # pos_ious = torch.sqrt(iou_targets.clone().detach() * pos_centerness_targets.clone().detach()[:,0]) + 1e-6
                
                cls_iou_targets[pos_inds, flatten_labels[pos_inds]] = iou_targets
                loss_cls = self.loss_cls(
                    flatten_cls_scores,
                    cls_iou_targets,
                    avg_factor=num_pos)
            else:     
                loss_cls = self.loss_cls(
                    flatten_cls_scores, flatten_labels, avg_factor=num_pos)
        else:
            if self.loss_cls.__str__() == 'VarifocalLoss()':
                flatten_labels = torch.zeros_like(flatten_cls_scores)
            loss_cls = self.loss_cls(
                flatten_cls_scores, flatten_labels, avg_factor=num_pos)
            loss_bbox = pos_bbox_preds.sum()

        return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox)
    
    def get_targets(self, points, gt_bboxes_list, gt_labels_list, 
                    bbox_preds, angle_preds, cls_scores,
                    debug = False,
                    matching = 0):

        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        
        '''labelassignment'''
        bn = bbox_preds[0].shape[0]
        rbbox_preds = [
            torch.cat([b.view(bn, 4, -1).permute(0,2,1), 
                       a.view(bn, 1, -1).permute(0,2,1)], dim=-1)
            for b, a in zip(bbox_preds, angle_preds)
        ]
        rbbox_probs = [torch.cat([p.view(bn, 15, -1).permute(0,2,1)], dim=-1) for p in cls_scores]
        concat_rbboxes = [preds for preds in torch.cat(rbbox_preds, dim=1)]
        concat_probs = [probs for probs in torch.cat(rbbox_probs, dim=1)]
        '''---------------'''
        
        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]
        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, angle_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            concat_rbboxes, ##
            concat_probs,
            points=concat_points,  ## 
            regress_ranges=concat_regress_ranges, ##
            num_points_per_lvl= num_points,
            debug = debug,
            matching = matching) ##
        
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
        # centerness_targets_list = [
        #     centerness_targets.split(num_points, 0)
        #     for centerness_targets in centerness_targets_list
        # ]
        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        concat_lvl_centerness_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            # centerness_targets = torch.cat(
            #     [centerness_targets[i] for centerness_targets in centerness_targets_list])
            bbox_targets = bbox_targets / self.featmap_strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
            # concat_lvl_centerness_targets.append(centerness_targets)
            
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_angle_targets)

    def _get_target_single(self, gt_bboxes, gt_labels, bbox_preds, probs, points, regress_ranges,
                           num_points_per_lvl, debug = False, matching = 0):
        """Compute regression, classification and angle targets for a single
        image."""
        RIoU = build_loss(dict(type='RotatedIoULoss', loss_weight=1.0,
                               reduction='none', mode='linear')) ##
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        annotation_bbox = gt_bboxes
        bboxes = bbox_preds.clone().detach() ##
        centerness_targets = []
        alpha = 0.5
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
        centerness_targets = 1 -torch.sqrt(((delta_x**2 + delta_y**2)+ 1e-8)/2)
        centerness_targets = centerness_targets.clamp_min(0)
        
        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        radius = self.center_sample_radius
        stride = offset.new_zeros(offset.shape)
        
        # OBB sampling region (9-2, 10-1, 10-2)
        if matching == 102:
            
            stride_obb = offset.new_zeros(offset.shape)
            ann_wh = annotation_bbox[:,2:4]
            vl, _ = ann_wh.sort(descending=True)
            ratio = vl[:,0]/vl[:,1]
            lvl_begin = 0
            stride_lvl =  torch.zeros_like(ann_wh)
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride_lvl[None] = self.featmap_strides[lvl_idx] * radius
                stride_lvl[torch.arange(stride_lvl.size(0)), ann_wh.argmax(dim=-1)] *= ratio
                stride_obb[lvl_begin:lvl_end] = stride_lvl
                lvl_begin = lvl_end
                
            inside_center_obb_mask = (abs(offset) < stride_obb).all(dim=-1)
            inside_gt_bbox_mask = torch.logical_and(inside_center_obb_mask,
                                                inside_gt_bbox_mask)
            
            det_probs = probs.softmax(-1)
            det_rbboxes = self.bbox_coder.decode(points[:,0], bboxes) #lrtba -> ctr wh a
            inside_topk_lvl = torch.zeros_like(inside_gt_bbox_mask)
            
            for i, gt in enumerate(annotation_bbox): # matching matrix init
                matching_score = centerness_targets[:,i].pow_(2) * \
                    (1-(RIoU(det_rbboxes, gt.expand(len(points), 5))*(10**3)).round()/ (10**3)).pow_(2) * \
                        det_probs[:, gt_labels[i]].rsqrt_() # matching matrix
                
                matching_score *= inside_gt_bbox_mask[:,i] # sampling region
                
                k = 15
                k_ = k//3
                ed = len(points)
                for num_points_lvl in reversed(num_points_per_lvl):
                    st = ed - num_points_lvl
                    # lvl_matrix = matching_score[st:ed]
                    act_num = (matching_score[st:ed] != 0).sum()
                    n = min(k_, act_num, k)
                    if act_num > 0:
                        arg_score_idxes = torch.argsort(matching_score[st:ed])[-n::] # target fmap의 matching score를 오름차순으로 정렬한뒤 n만큼 추출
                        inside_topk_lvl[st:ed,i][arg_score_idxes] = True
                        k -= n
                    k_ += k_ - n 
                    k_ = min(k_, k)
                    ed = st
            
            areas *= inside_gt_bbox_mask
            areas *= inside_topk_lvl
            
            max_area, max_area_inds = areas.max(dim=1) # 겹치는 points를 늘린 후 max값 선택
            if 0 in max_area_inds.bincount().tolist(): # 0개의 anchorpoint가 할당된 경우
                noacp = torch.where(max_area_inds[max_area != 0].bincount() == 0)[0] # GT 중 match가 안된 gt ind
                max_area_inds[centerness_targets.argmax(dim=0)[noacp]] = noacp # ctr기반으로 할당
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]
            return labels, bbox_targets, angle_targets
        
        if matching == 10:
            
            stride_obb = offset.new_zeros(offset.shape)
            ann_wh = annotation_bbox[:,2:4]
            vl, _ = ann_wh.sort(descending=True)
            ratio = vl[:,0]/vl[:,1]
            lvl_begin = 0
            stride_lvl =  torch.zeros_like(ann_wh)
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride_lvl[None] = self.featmap_strides[lvl_idx] * radius
                stride_lvl[torch.arange(stride_lvl.size(0)), ann_wh.argmax(dim=-1)] *= ratio
                stride_obb[lvl_begin:lvl_end] = stride_lvl
                lvl_begin = lvl_end
                
            inside_center_obb_mask = (abs(offset) < stride_obb).all(dim=-1)
            inside_gt_bbox_mask = torch.logical_and(inside_center_obb_mask,
                                                inside_gt_bbox_mask)
            areas *= inside_gt_bbox_mask
                        
            det_probs = probs.clone().detach().softmax(-1)
            det_rbboxes = self.bbox_coder.decode(points[:,0], bboxes) #lrtba -> ctr wh a
            inside_topk_lvl = torch.zeros_like(inside_gt_bbox_mask).float()
            
            for i, gt in enumerate(annotation_bbox): # matching matrix init
                matching_score = centerness_targets[:,i].pow_(2) *\
                        (1-(RIoU(det_rbboxes, gt.expand(len(points), 5))*(10**3)).round()/ (10**3)).pow_(2) *\
                            det_probs[:, gt_labels[i]].rsqrt_() # matching matrix

                
                areas[:,i] = matching_score
                matching_score *= inside_gt_bbox_mask[:,i]
                k = 15
                ed = len(points)
                for num_points_lvl in reversed(num_points_per_lvl):
                    if k > 0:
                        # lvl_rvs_idx = 2-lvl_idx
                        st = ed - num_points_lvl
                        lvl_matrix = matching_score[st:ed]
                        act_num = (lvl_matrix != 0).sum()
                        if act_num > 0:
                            arg_score_idxes = torch.argsort(lvl_matrix)[-min(k,act_num)::]
                            # areas[ingt_idx] = matrix
                            inside_topk_lvl[st:ed,i][arg_score_idxes] = lvl_matrix[arg_score_idxes]
                            k -= min(k, act_num)
                        ed = st
                    else:
                        break
            areas *= inside_topk_lvl
            
            max_area, max_area_inds = areas.max(dim=1) # 겹치는 points를 늘린 후 max값 선택
            if 0 in max_area_inds.bincount().tolist(): # 0개의 anchorpoint가 할당된 경우
                noacp = torch.where(max_area_inds[max_area != 0].bincount() == 0)[0] # GT 중 match가 안된 gt ind
                max_area_inds[centerness_targets.argmax(dim=0)[noacp]] = noacp # ctr기반으로 할당
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]
            
            return labels, bbox_targets, angle_targets
        
        # condition3: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))
        
        if matching == 92:
            stride_obb = offset.new_zeros(offset.shape)
            ann_wh = annotation_bbox[:,2:4]
            vl, _ = ann_wh.sort(descending=True)
            ratio = vl[:,0]/vl[:,1]
            lvl_begin = 0
            stride_lvl =  torch.zeros_like(ann_wh)
            
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride_lvl[None] = self.featmap_strides[lvl_idx] * radius
                stride_lvl[torch.arange(stride_lvl.size(0)), ann_wh.argmax(dim=-1)] *= ratio
                stride_obb[lvl_begin:lvl_end] = stride_lvl
                lvl_begin = lvl_end
                
            inside_center_obb_mask = (abs(offset) < stride_obb).all(dim=-1)
        
            #condition2 : sampling region
            inside_gt_bbox_mask = torch.logical_and(inside_center_obb_mask,
                                                inside_gt_bbox_mask)
            
            areas *= inside_gt_bbox_mask
            areas *= inside_regress_range
            
            det_probs = probs.clone().detach().softmax(-1)
            det_rbboxes = self.bbox_coder.decode(points[:,0], bboxes) #lrtba -> ctr wh a
            inside_topk_lvl = torch.zeros_like(inside_gt_bbox_mask).float()
            for i, gt in enumerate(annotation_bbox):
                matching_cost = centerness_targets[:,i].pow_(2) *\
                    (1-(RIoU(det_rbboxes, gt.expand(len(points), 5))*(10**3)).round()/ (10**3)).pow_(2) *\
                        det_probs[:, gt_labels[i]].rsqrt_() # matching matrix

                matching_cost *= inside_gt_bbox_mask[:,i]
                matching_cost *= inside_regress_range[:,i]
                                
                if matching_cost.shape[0] >= 15:
                    k = 15
                else:
                    k = matching_cost.shape[0]
                    
                val, idx = matching_cost.sort(descending=True)
                idx = idx[:k]
                val = val[:k]
                inside_topk_lvl[idx,i] = val # cost가 0인 값은 False가 되도록
                
            areas *= inside_topk_lvl.bool()
            
            max_area, max_area_inds = areas.max(dim=1)
            if 0 in max_area_inds.bincount().tolist(): # 0개의 anchorpoint가 할당된 경우
                noacp = torch.where(max_area_inds[max_area != 0].bincount() == 0)[0] # GT 중 match가 안된 gt ind
                max_area_inds[centerness_targets.argmax(dim=0)[noacp]] = noacp # ctr기반으로 할당
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]
            return labels, bbox_targets, angle_targets
        
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
        
        if matching == 103:
            areas *= inside_gt_bbox_mask
                        
            det_probs = probs.clone().detach().softmax(-1)
            det_rbboxes = self.bbox_coder.decode(points[:,0], bboxes) #lrtba -> ctr wh a
            inside_topk_lvl = torch.zeros_like(inside_gt_bbox_mask).float()
            
            for i, gt in enumerate(annotation_bbox): # matching matrix init
                matching_score = centerness_targets[:,i].pow_(2) *\
                        (1-(RIoU(det_rbboxes, gt.expand(len(points), 5))*(10**3)).round()/ (10**3)).pow_(2) *\
                            det_probs[:, gt_labels[i]].rsqrt_() # matching matrix

                
                areas[:,i] = matching_score
                matching_score *= inside_gt_bbox_mask[:,i]
                k = 15
                ed = len(points)
                for num_points_lvl in reversed(num_points_per_lvl):
                    if k > 0:
                        # lvl_rvs_idx = 2-lvl_idx
                        st = ed - num_points_lvl
                        lvl_matrix = matching_score[st:ed]
                        act_num = (lvl_matrix != 0).sum()
                        if act_num > 0:
                            arg_score_idxes = torch.argsort(lvl_matrix)[-min(k,act_num)::]
                            # areas[ingt_idx] = matrix
                            inside_topk_lvl[st:ed,i][arg_score_idxes] = lvl_matrix[arg_score_idxes]
                            k -= min(k, act_num)
                        ed = st
                    else:
                        break
            areas *= inside_topk_lvl
            
            max_area, max_area_inds = areas.max(dim=1) # 겹치는 points를 늘린 후 max값 선택
            if 0 in max_area_inds.bincount().tolist(): # 0개의 anchorpoint가 할당된 경우
                noacp = torch.where(max_area_inds[max_area != 0].bincount() == 0)[0] # GT 중 match가 안된 gt ind
                max_area_inds[centerness_targets.argmax(dim=0)[noacp]] = noacp # ctr기반으로 할당
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]
            
            return labels, bbox_targets, angle_targets
        
        if matching == 104:
            areas *= inside_gt_bbox_mask
                        
            det_probs = probs.clone().detach().softmax(-1)
            det_rbboxes = self.bbox_coder.decode(points[:,0], bboxes) #lrtba -> ctr wh a
            inside_topk_lvl = torch.zeros_like(inside_gt_bbox_mask).float()
            
            for i, gt in enumerate(annotation_bbox): # matching matrix init
                matching_score = centerness_targets[:,i].pow_(2) * \
                    (1-(RIoU(det_rbboxes, gt.expand(len(points), 5))*(10**3)).round()/ (10**3)).pow_(2) * \
                        det_probs[:, gt_labels[i]].rsqrt_() # matching matrix
                
                matching_score *= inside_gt_bbox_mask[:,i] # sampling region
                
                k = 15
                k_ = k//3
                ed = len(points)
                for num_points_lvl in reversed(num_points_per_lvl):
                    st = ed - num_points_lvl
                    # lvl_matrix = matching_score[st:ed]
                    act_num = (matching_score[st:ed] != 0).sum()
                    n = min(k_, act_num, k)
                    if act_num > 0:
                        arg_score_idxes = torch.argsort(matching_score[st:ed])[-n::] # target fmap의 matching score를 오름차순으로 정렬한뒤 n만큼 추출
                        inside_topk_lvl[st:ed,i][arg_score_idxes] = True
                        k -= n
                    k_ += k_ - n 
                    k_ = min(k_, k)
                    ed = st
            
            areas *= inside_topk_lvl
            
            max_area, max_area_inds = areas.max(dim=1) # 겹치는 points를 늘린 후 max값 선택
            if 0 in max_area_inds.bincount().tolist(): # 0개의 anchorpoint가 할당된 경우
                noacp = torch.where(max_area_inds[max_area != 0].bincount() == 0)[0] # GT 중 match가 안된 gt ind
                max_area_inds[centerness_targets.argmax(dim=0)[noacp]] = noacp # ctr기반으로 할당
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]
            
            return labels, bbox_targets, angle_targets
            
        if matching == 91:
            areas *= inside_gt_bbox_mask ## 1
            areas *= inside_regress_range ## 2
            
            # ingt_idx = torch.where(areas!=0)[0] # inside of GT bbox indicator
            det_probs = probs.softmax(-1)
            # matrix = torch.zeros(len(points[ingt_idx]), len(annotation_bbox), device=areas.device) 
            det_rbboxes = self.bbox_coder.decode(points[:,0], bboxes) #lrtba -> ctr wh a
            inside_topk_lvl = torch.zeros_like(inside_gt_bbox_mask).float()
            for i, gt in enumerate(annotation_bbox):
                matching_cost = centerness_targets[:,i].pow_(2) *\
                    (1-(RIoU(det_rbboxes, gt.expand(len(points), 5))*(10**3)).round()/ (10**3)).pow_(2) *\
                        det_probs[:, gt_labels[i]].rsqrt_() # matching matrix

                matching_cost *= inside_gt_bbox_mask[:,i]
                matching_cost *= inside_regress_range[:,i]
                
                if matching_cost.shape[0] >= 15:
                    k = 15
                else:
                    k = matching_cost.shape[0]
                    
                val, idx = matching_cost.sort(descending=True)
                idx = idx[:k]
                val = val[:k]
                inside_topk_lvl[idx,i] = val # cost가 0인 값은 False가 되도록
                
            areas *= inside_topk_lvl.bool()
                
            max_area, max_area_inds = areas.max(dim=1) # 겹치는 points를 늘린 후 max값 선택
            
            if 0 in max_area_inds.bincount().tolist(): # 0개의 anchorpoint가 할당된 경우
                noacp = torch.where(max_area_inds[max_area != 0].bincount() == 0)[0] # GT 중 match가 안된 gt ind
                max_area_inds[centerness_targets.argmax(dim=0)[noacp]] = noacp # ctr기반으로 할당
            
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]
            
            return labels, bbox_targets, angle_targets
        
        if matching == 11:
            areas *= inside_gt_bbox_mask ## 1
            det_probs = probs.softmax(-1)
            det_rbboxes = self.bbox_coder.decode(points[:,0], bboxes) #lrtba -> ctr wh a
            inside_topk_lvl = torch.zeros_like(inside_gt_bbox_mask).float()

            for i, gt in enumerate(annotation_bbox):
                matching_cost = 0.2 * centerness_targets[:,i] +\
                                0.2 * (1-(RIoU(det_rbboxes, gt.expand(len(points), 5))*(10**3)).round()/ (10**3)) +\
                                0.6 * det_probs[:, gt_labels[i]]

                # matching_cost = centerness_targets[:,i].pow_(2) *\
                #     (1-(RIoU(det_rbboxes, gt.expand(len(points), 5))*(10**3)).round()/ (10**3)).pow_(2)*\
                #         det_probs[:, gt_labels[i]].rsqrt_() # matching matrix
                
                #  (1-(RIoU(det_rbboxes, gt.expand(len(points), 5))*(10**3)).round()/ (10**3)).pow_(2)
                # (((1-RIoU(det_rbboxes, gt.expand(len(points), 5)))*(10**3)).round()/(10**3)).pow_(2)
                matching_cost *= inside_gt_bbox_mask[:,i]
                # matching_cost *= inside_regress_range[:,i]
                
                if matching_cost.shape[0] >= 15:
                    k = 15
                else:
                    k = matching_cost.shape[0]
                    
                val, idx = matching_cost.sort(descending=True)
                idx = idx[:k]
                val = val[:k]
                inside_topk_lvl[idx,i] = val # cost가 0인 값은 False가 되도록

            areas *= inside_topk_lvl.bool()
                
            max_area, max_area_inds = areas.max(dim=1) # 겹치는 points를 늘린 후 max값 선택
            
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            
            if 0 in max_area_inds.bincount().tolist(): # 0개의 anchorpoint가 할당된 경우
                noacp = torch.where(max_area_inds[max_area != 0].bincount() == 0)[0] # GT 중 match가 안된 gt ind
                rematchig_anchor = centerness_targets.argmax(dim=0)[noacp]
                max_area_inds[rematchig_anchor] = noacp # ctr기반으로 할당
                labels[rematchig_anchor] = gt_labels[noacp]
                # print(centerness_targets.argmax(dim=0)[noacp])
                
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]
        
        if matching == 81:
        
            areas *= inside_gt_bbox_mask
            ingt_idx = torch.where(areas!=0)[0] # inside of GT bbox indicator
            det_probs = probs[ingt_idx].softmax(-1)
            matrix = torch.zeros(len(points[ingt_idx]), len(annotation_bbox), device=areas.device) 
            det_rbboxes = self.bbox_coder.decode(points[ingt_idx][:,0], bboxes[ingt_idx]) #lrtba -> ctr wh a
            
            if len(ingt_idx) != 0: 
                for i, gt in enumerate(annotation_bbox):
                    gt = gt.expand(len(ingt_idx), 5)
                    cls = gt_labels[i]
                    prob = det_probs[:, cls] 
                    try:
                        iou =  1-(RIoU(det_rbboxes, gt)*(10**3)).round()/ (10**3)
                    except:
                        break
                    matching_cost =  iou * prob # matching matrix
                    if matching_cost.shape[0] >= 15:
                        k = 15
                    else:
                        k = matching_cost.shape[0]
                    val, idx = torch.topk(matching_cost, k=k, dim=0) # top k=15  
                    matrix[idx[val!=0],i] = val[val!=0] # remove 0
            
                areas[ingt_idx] = matrix
                    
            max_area, max_area_inds = areas.max(dim=1) # 겹치는 points를 늘린 후 max값 선택
            
            if 0 in max_area_inds.bincount().tolist(): # 0개의 anchorpoint가 할당된 경우
                noacp = torch.where(max_area_inds[max_area != 0].bincount() == 0)[0] # GT 중 match가 안된 gt ind
                max_area_inds[centerness_targets.argmax(dim=0)[noacp]] = noacp # ctr기반으로 할당
            
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]
            
            return labels, bbox_targets, angle_targets

        if matching == 82:
            areas *= inside_regress_range
            areas *= inside_gt_bbox_mask
            ingt_idx = torch.where(areas!=0)[0] # inside of GT bbox indicator
            
            det_probs = probs[ingt_idx].softmax(-1)
            matrix = torch.zeros(len(points[ingt_idx]), len(annotation_bbox), device=areas.device) 
            det_rbboxes = self.bbox_coder.decode(points[ingt_idx][:,0], bboxes[ingt_idx]) #lrtba -> ctr wh a
            
            if len(ingt_idx) != 0: 
                for i, gt in enumerate(annotation_bbox):
                    gt = gt.expand(len(ingt_idx), 5)
                    cls = gt_labels[i]
                    prob = det_probs[:, cls] 
                    try:
                        iou =  1-(RIoU(det_rbboxes, gt)*(10**3)).round()/ (10**3)
                    except:
                        break
                    matching_cost =  iou * prob # matching matrix
                    if matching_cost.shape[0] >= 15:
                        k = 15
                    else:
                        k = matching_cost.shape[0]
                    val, idx = torch.topk(matching_cost, k=k, dim=0) # top k=15  
                    matrix[idx[val!=0],i] = val[val!=0] # remove 0
            
                areas[ingt_idx] = matrix
                
            max_area, max_area_inds = areas.max(dim=1) # 겹치는 points를 늘린 후 max값 선택
            
            if 0 in max_area_inds.bincount().tolist(): # 0개의 anchorpoint가 할당된 경우
                noacp = torch.where(max_area_inds[max_area != 0].bincount() == 0)[0] # GT 중 match가 안된 gt ind
                max_area_inds[centerness_targets.argmax(dim=0)[noacp]] = noacp # ctr기반으로 할당
            
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]      
            
        if matching == 0:
            
            '''
            Default matching
            '''
            
            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            areas[inside_gt_bbox_mask == 0] = INF ## (1,2)
            areas[inside_regress_range == 0] = INF ## (3)
            min_area, min_area_inds = areas.min(dim=1)
            labels = gt_labels[min_area_inds] 
            labels[min_area == INF] = self.num_classes  # set as BG
            bbox_targets = bbox_targets[range(num_points), min_area_inds]
            angle_targets = gt_angle[range(num_points), min_area_inds]

        elif matching == 1:

            '''
            Rotated IoU 기반 matching ver.1
            '''
            
            areas[inside_gt_bbox_mask == 0] = 0 ## (1,2)
            areas[inside_regress_range == 0] = 0 ## (3)
            overlap_idx = torch.where(torch.bincount(torch.where(areas != 0)[0]) >= 2)[0] # 위에서 2개이상의 bbox가 할당된 det 인덱스
            if len(overlap_idx) != 0:
                pts = points[overlap_idx]
                iou_matrix = torch.zeros(len(overlap_idx), len(annotation_bbox), device=areas.device) 
                det_rbboxes = self.bbox_coder.decode(pts[:,0], bboxes[overlap_idx])
                mask = (areas[overlap_idx] != 0).float()
                for i, gt in enumerate(annotation_bbox):
                    gt = gt.expand(len(overlap_idx), 5)
                    iou =  1-(RIoU(det_rbboxes, gt)*(10**3)).round()/ (10**3)
                    iou_matrix[:,i] = iou
                iou_matrix *= mask
                areas[overlap_idx] = iou_matrix
            max_area, max_area_inds = areas.max(dim=1) # 더 큰 IoU 선택
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]

        elif matching == 2 :
            
            '''
            Rotated IoU 기반 matching ver.2
            '''
            areas[inside_gt_bbox_mask == 0] = 0 ## (1,2)
            areas[inside_regress_range == 0] = 0 ## (3)
            overlap_idx = torch.where(torch.bincount(torch.where(areas != 0)[0]) >= 2)[0] 
            
            if len(overlap_idx) != 0:
                pts = points[overlap_idx]
                iou_matrix = torch.zeros(len(overlap_idx), len(annotation_bbox), device=areas.device) 
                det_rbboxes = self.bbox_coder.decode(pts[:,0], bboxes[overlap_idx])
                mask = (areas[overlap_idx] != 0).float()
                for i, gt in enumerate(annotation_bbox):
                    gt = gt.expand(len(overlap_idx), 5)
                    iou =  1-(RIoU(det_rbboxes, gt)*(10**3)).round()/ (10**3)
                    iou_matrix[:,i] = iou
                iou_matrix *= mask
                areas[overlap_idx] = iou_matrix
            max_area, max_area_inds = areas.max(dim=1) # 더 큰 IoU 선택
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
                
            if not torch.equal(labels.unique()[:-1],gt_labels.unique()): # BG제외 label 할당 안 된 class가 있으면 (하나의 이미지에 여러 gt가 있을때.. 문제 생길 수 있음 ㅠ)
                # 할당 안 된 라벨과 gt bbox 찾기
                not_assigned = gt_labels.unique()[~(gt_labels.unique()[..., None] == labels.unique()[:-1]).any(-1)]
                for na in not_assigned:
                    dist = linalg.vector_norm(offset[:,gt_labels==na],dim=-1) # anchor point와 gt center와의 거리
                    _, assign_idx = torch.topk(-dist, k=3, dim=0) # 각 gt별 가장 가까운 anchor point (1,2에서 filtering되었더라도 여기서 추가될 수도 있음)                    
                    areas[assign_idx] = INF # 할당이 안됐을때, GT랑 가장 가까운 3개의 포인트는 무조건 할당 (l,r,t,b가 음수가 될 수도 있는데 문제 없나?)
                
            max_area, max_area_inds = areas.max(dim=1) # 더 큰 IoU 선택
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]
            
        elif matching == 3 :
            
            '''
            Rotated IoU 기반 matching ver.3
            '''
            areas[inside_gt_bbox_mask == 0] = 0 ## (1,2)
            # areas[inside_regress_range == 0] = 0 ## (3)
            overlap_idx = torch.where(torch.bincount(torch.where(areas != 0)[0]) >= 2)[0] 
            
            if len(overlap_idx) != 0:
                pts = points[overlap_idx]
                iou_matrix = torch.zeros(len(overlap_idx), len(annotation_bbox), device=areas.device) 
                det_rbboxes = self.bbox_coder.decode(pts[:,0], bboxes[overlap_idx])
                mask = (areas[overlap_idx] != 0).float()
                for i, gt in enumerate(annotation_bbox):
                    gt = gt.expand(len(overlap_idx), 5)
                    iou =  1-(RIoU(det_rbboxes, gt)*(10**3)).round()/ (10**3)
                    iou_matrix[:,i] = iou
                iou_matrix *= mask
                areas[overlap_idx] = iou_matrix
                
            max_area, max_area_inds = areas.max(dim=1) # 더 큰 IoU 선택
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
                
            if not torch.equal(labels.unique()[:-1],gt_labels.unique()): # BG제외 label 할당 안 된 class가 있으면 (하나의 이미지에 여러 gt가 있을때.. 문제 생길 수 있음 ㅠ)
                # 할당 안 된 라벨과 gt bbox 찾기
                not_assigned = gt_labels.unique()[~(gt_labels.unique()[..., None] == labels.unique()[:-1]).any(-1)]
                for na in not_assigned:
                    dist = linalg.vector_norm(offset[:,gt_labels==na],dim=-1) # anchor point와 gt center와의 거리
                    _, assign_idx = torch.topk(-dist, k=3, dim=0) # 각 gt별 가장 가까운 anchor point (1,2에서 filtering되었더라도 여기서 추가될 수도 있음)                    
                    areas[assign_idx] = INF # 할당이 안됐을때, GT랑 가장 가까운 3개의 포인트는 무조건 할당 (l,r,t,b가 음수가 될 수도 있는데 문제 없나?)
                
            max_area, max_area_inds = areas.max(dim=1) # 더 큰 IoU 선택
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]
            
        elif matching == 4 :
            areas[inside_gt_bbox_mask == 0] = 0 ## (1,2)
            areas[inside_regress_range == 0] = 0 ## (3)
            overlap_idx = torch.where(torch.bincount(torch.where(areas != 0)[0]) >= 2)[0] 
            
            if len(overlap_idx) != 0:
                pts = points[overlap_idx]
                iou_matrix = torch.zeros(len(overlap_idx), len(annotation_bbox), device=areas.device) 
                det_overlapbboxes = self.bbox_coder.decode(pts[:,0], bboxes[overlap_idx])
                mask = (areas[overlap_idx] != 0).float()
                for i, gt in enumerate(annotation_bbox):
                    gt = gt.expand(len(overlap_idx), 5)
                    iou =  1-(RIoU(det_overlapbboxes, gt)*(10**3)).round()/ (10**3)
                    iou_matrix[:,i] = iou
                iou_matrix *= mask
                not_assigned =  list(set(torch.where(iou_matrix!=0)[1].tolist()) - set(iou_matrix.max(dim=-1)[1].tolist()))
                if len(not_assigned) != 0:
                    for i, iou_det in enumerate(iou_matrix):
                        if random.random() >= 0.5:
                            overlap_gtidx = torch.where(iou_det != 0)[0]
                            iou_matrix[i][overlap_gtidx] = iou_det[torch.flip(overlap_gtidx, dims=(0,))]
                            
                areas[overlap_idx] = iou_matrix
            max_area, max_area_inds = areas.max(dim=1) # 더 큰 IoU 선택
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]
            
        elif matching == 5:
            alpha = 0.5
            areas[inside_gt_bbox_mask == 0] = 0 ## (1,2)
            areas[inside_regress_range == 0] = 0 ## (3)
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
            max_area, max_area_inds = areas.max(dim=1) # 더 큰 IoU 선택
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]
        
        elif matching == 6:
            
            areas[inside_gt_bbox_mask == 0] = 0 ## (1,2)
            # areas[inside_regress_range == 0] = 0 ## (3)
            overlap_idx = torch.where(torch.bincount(torch.where(areas != 0)[0]) >= 2)[0] # 위에서 2개이상의 bbox가 할당된 det 인덱스
            if len(overlap_idx) != 0:
                pts = points[overlap_idx]
                iou_matrix = torch.zeros(len(overlap_idx), len(annotation_bbox), device=areas.device) 
                det_overlapbboxes = self.bbox_coder.decode(pts[:,0], bboxes[overlap_idx])
                det_probs = probs[overlap_idx].softmax(-1)
                mask = (areas[overlap_idx] != 0).float()
                for i, gt in enumerate(annotation_bbox):
                    
                    gt = gt.expand(len(overlap_idx), 5)
                    iou =  1-(RIoU(det_overlapbboxes, gt)*(10**3)).round()/ (10**3)
                    cls = gt_labels[i]
                    prob = det_probs[:, cls]
                    iou_matrix[:,i] = alpha * iou + (1-alpha) * prob
                iou_matrix *= mask
                areas[overlap_idx] = iou_matrix
            max_area, max_area_inds = areas.max(dim=1) # 더 큰 IoU 선택
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]
        
        elif matching == 7:
            areas[inside_gt_bbox_mask == 0] = 0 ## (1,2)
            # areas[inside_regress_range == 0] = 0 ## (3)
            overlap_idx = torch.where(torch.bincount(torch.where(areas != 0)[0]) >= 2)[0] 
            
            if len(overlap_idx) != 0:
                pts = points[overlap_idx]
                iou_matrix = torch.zeros(len(overlap_idx), len(annotation_bbox), device=areas.device) 
                det_overlapbboxes = self.bbox_coder.decode(pts[:,0], bboxes[overlap_idx])
                mask = (areas[overlap_idx] != 0).float()
                for i, gt in enumerate(annotation_bbox):
                    gt = gt.expand(len(overlap_idx), 5)
                    iou =  1-(RIoU(det_overlapbboxes, gt)*(10**3)).round()/ (10**3)
                    iou_matrix[:,i] = iou
                iou_matrix *= mask
                not_assigned =  list(set(torch.where(iou_matrix!=0)[1].tolist()) - set(iou_matrix.max(dim=-1)[1].tolist()))
                if len(not_assigned) != 0:
                    for i, iou_det in enumerate(iou_matrix):
                        if random.random() >= 0.5:
                            overlap_gtidx = torch.where(iou_det != 0)[0]
                            iou_matrix[i][overlap_gtidx] = iou_det[torch.flip(overlap_gtidx, dims=(0,))]

                areas[overlap_idx] = iou_matrix
            max_area, max_area_inds = areas.max(dim=1) # 더 큰 IoU 선택
            labels = gt_labels[max_area_inds]
            labels[max_area == 0] = self.num_classes
            bbox_targets = bbox_targets[range(num_points), max_area_inds]
            angle_targets = gt_angle[range(num_points), max_area_inds]
            
        if debug:
            ig = cv2.imread(self.img_metas['filename'])
            ig = cv2.cvtColor(ig, cv2.COLOR_BGR2RGB)
            for i, lb in enumerate(labels):
                key = str(lb.item())
                x = points[i][0][0].detach().item()
                y = points[i][0][1].detach().item()
                if key == '15':
                    cv2.circle(ig, (int(x), int(y)), 1, (0, 0, 0), 0, 1)
                    
                else:
                    cv2.circle(ig,  (int(x), int(y)), 3, PALETTE[int(key)], -1, 1)
                
            for i, p in enumerate(PALETTE):
                cv2.putText(ig, CLASSES[i], (20, 20*i+10), 1, 1, p)

            cv2.imwrite(f'data/debugging/la{matching}/'+ self.img_metas['ori_filename'], ig)
    
        return labels, bbox_targets, angle_targets
    
    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   angle_preds,
                   img_metas,
                   cfg=None,
                   rescale=None):
        
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            angle_pred_list = [
                angle_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 angle_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           angle_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, angle_pred, points in zip(
                cls_scores, bbox_preds, angle_preds, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img)
        return det_bboxes, det_labels

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def refine_bboxes(self, cls_scores, bbox_preds, angle_preds):
        """This function will be used in S2ANet, whose num_anchors=1."""
        num_levels = len(cls_scores)
        assert num_levels == len(bbox_preds)
        num_imgs = cls_scores[0].size(0)
        for i in range(num_levels):
            assert num_imgs == cls_scores[i].size(0) == bbox_preds[i].size(0)

        # device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
        bboxes_list = [[] for _ in range(num_imgs)]

        for lvl in range(num_levels):
            bbox_pred = bbox_preds[lvl]
            angle_pred = angle_preds[lvl]
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred.reshape(num_imgs, -1, 4)
            angle_pred = angle_pred.permute(0, 2, 3, 1)
            angle_pred = angle_pred.reshape(num_imgs, -1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=-1)

            points = mlvl_points[lvl]

            for img_id in range(num_imgs):
                bbox_pred_i = bbox_pred[img_id]
                decode_bbox_i = self.bbox_coder.decode(points, bbox_pred_i)
                bboxes_list[img_id].append(decode_bbox_i.detach())

        return bboxes_list

'''
@ROTATED_HEADS.register_module()
class RotatedYOLOv8Head(RotatedYOLOv6Head):
    """YOLOv8 Head"""
    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 reg_max = 0,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 regress_ranges=((-1, 64), (64, 128), (128, 256)),
                 bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 matching : int = 0,
                 debug = False, 
                #  loss_angle=dict(type='L1Loss', loss_weight=1.0),
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
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.reg_max = reg_max
        self.matching = matching
        self.debug = debug
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        super().__init__(num_classes, in_channels, widen_factor, reg_max, featmap_strides,
                         regress_ranges=regress_ranges, matching = matching, debug=debug,
                         bbox_coder=bbox_coder, loss_cls=loss_cls, loss_bbox=loss_bbox, norm_cfg=norm_cfg,
                         act_cfg=act_cfg, init_cfg=init_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
        
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.prior_generator = MlvlPointGenerator(featmap_strides)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        
    
    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        # super().init_weights()
        for reg_pred, cls_pred, ang_pred, stride in zip(self.reg_preds, self.cls_preds, self.ang_preds,
                                              self.featmap_strides):
            reg_pred[-1].bias.data[:] = 1.0  # box
            ang_pred[-1].bias.data[:] = 1.0  # angle
            # cls (.01 objects, 80 classes, 640 img)
            cls_pred[-1].bias.data[:self.num_classes] = math.log(
                5 / self.num_classes / (640 / stride)**2)

    def _init_layers(self):
        
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.ang_preds = nn.ModuleList()
        
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.featmap_strides])
        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=reg_out_channels,
                              out_channels=(self.num_base_priors + self.reg_max) * 4,
                              kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.num_classes,
                              kernel_size=1)))
            self.ang_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=(self.num_base_priors + self.reg_max) * 1,
                              kernel_size=1)))

        if self.reg_max > 1:
            proj = torch.arange(
                self.reg_max + self.num_base_priors, dtype=torch.float)
            self.register_buffer('proj', proj, persistent=False)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions.
        """
        # x = [x[-3]]
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.cls_preds, 
                           self.reg_preds, self.ang_preds, self.scales, self.featmap_strides)

    def forward_single(self, x: Tensor, cls_pred: nn.Module, 
                       reg_pred: nn.Module,  ang_pred: nn.Module, 
                       scale: List, stride) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        cls_logit = cls_pred(x)
        bbox_dist_preds = reg_pred(x)
        predicted_angle = ang_pred(x)

        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max + self.num_base_priors,
                 h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = scale(bbox_dist_preds).float()
            bbox_preds = bbox_preds.clamp(min=0)
            if not self.training:
                bbox_preds *= stride
        if self.training:
            return cls_logit, bbox_preds, predicted_angle
        else:
            return cls_logit, bbox_preds, predicted_angle
'''

class ContrastiveHead(BaseModule):
    """Contrastive Head for YOLO-World
    compute the region-text scores according to the
    similarity between image and text features
    Args:
        embed_dims (int): embed dim of text and image features
    """

    def __init__(self,
                 embed_dims: int,
                 init_cfg: OptConfigType = None,
                 use_einsum: bool = False) -> None:

        super().__init__(init_cfg=init_cfg)

        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)

        if self.use_einsum:
            x = torch.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)  # bchw->bhwc
            x = x.reshape(batch, -1, channel)  # bhwc->b(hw)c
            w = w.permute(0, 2, 1)  # bkc->bck
            x = torch.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x

class RepBNContrastiveHead(BaseModule):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    Args:
        embed_dims (int): embed dim of text and image features
        norm_cfg (dict): normalization params
    """

    def __init__(self,
                 embed_dims: int,
                 num_guide_embeds: int,
                 norm_cfg,
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.conv = nn.Conv2d(embed_dims, num_guide_embeds, kernel_size=1)
        self.bias = nn.Parameter(torch.zeros([]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        
        x = self.conv(x)
        x = F.normalize(x, dim=-1, p=2)
        x = x * self.logit_scale.exp() + self.bias
        
        
        return x


class BNContrastiveHead(BaseModule):
    """ Batch Norm Contrastive Head for YOLO-World
    using batch norm instead of l2-normalization
    Args:
        embed_dims (int): embed dim of text and image features
        norm_cfg (dict): normalization params
    """

    def __init__(self,
                 embed_dims: int,
                 norm_cfg,
                 init_cfg: OptConfigType = None,
                 use_einsum: bool = True) -> None:

        super().__init__(init_cfg=init_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        self.bias = nn.Parameter(torch.zeros([]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))
        self.use_einsum = use_einsum

    def forward(self, x: Tensor, w: Tensor) -> Tensor:
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)

        if self.use_einsum:
            x = torch.einsum('bchw,bkc->bkhw', x, w)
        else:
            batch, channel, height, width = x.shape
            _, k, _ = w.shape
            x = x.permute(0, 2, 3, 1)  # bchw->bhwc
            x = x.reshape(batch, -1, channel)  # bhwc->b(hw)c
            w = w.permute(0, 2, 1)  # bkc->bck
            x = torch.matmul(x, w)
            x = x.reshape(batch, height, width, k)
            x = x.permute(0, 3, 1, 2)

        x = x * self.logit_scale.exp() + self.bias
        return x
    

@ROTATED_HEADS.register_module()
class RotatedYOLOWorldHead(RotatedYOLOv6Head):
    """YOLO-World Head
    """

    def __init__(self,
                 reparameterized : bool,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 contrast_softmax = False,
                 reg_max = 0,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                #  loss_angle=dict(type='L1Loss', loss_weight=1.0),
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
                 test_cfg=None,
                 embed_dims = 512,
                 num_guide = 15,
                 use_bn_head = False,
                 freeze_all = False,
                 world_size = -1):
        self.reparameterized = reparameterized
        self.freeze_all = freeze_all
        self.embed_dims = embed_dims
        self.world_size = world_size
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.reg_max = reg_max
        self.num_guide = num_guide
        self.use_bn_head = use_bn_head
        self.contrast_softmax = contrast_softmax
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        super().__init__(num_classes, in_channels, widen_factor, reg_max, featmap_strides,
                         bbox_coder=bbox_coder, loss_cls=loss_cls, loss_bbox=loss_bbox, norm_cfg=norm_cfg,
                         act_cfg=act_cfg, init_cfg=init_cfg, train_cfg=train_cfg, test_cfg=test_cfg)
        
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.prior_generator = MlvlPointGenerator(featmap_strides)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        # self.assigner = build_assigner(train_cfg.assigner)
        # self._init_layers()

    # def _init_layers(self):
    #     """initialize conv layers in YOLOv6 head."""
    #     # Init decouple head
    #     self.cls_preds = nn.ModuleList()
    #     self.reg_preds = nn.ModuleList()
    #     self.ang_preds = nn.ModuleList()
    #     self.scales = nn.ModuleList([Scale(1.0) for _ in self.featmap_strides])
    #     self.cls_contrasts = nn.ModuleList()

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        # super().init_weights()
        for cls_pred, cls_contrast, stride in zip(self.cls_preds,
                                                  self.cls_contrasts,
                                                  self.featmap_strides):
            cls_pred[-1].bias.data[:] = 0.0  # reset bias
            if hasattr(cls_contrast, 'bias'):
                nn.init.constant_(
                    cls_contrast.bias.data,
                    math.log(5 / self.num_classes / (640 / stride)**2))

    def _init_layers(self) -> None:
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.ang_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.featmap_strides])
        reg_out_channels = max(
            (16, self.in_channels[0] // 4, 16 * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=reg_out_channels,
                               out_channels=reg_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=reg_out_channels,
                              out_channels=(self.num_base_priors + self.reg_max) * 4,
                              kernel_size=1)))
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.embed_dims,
                              kernel_size=1)))
            self.ang_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=self.in_channels[i],
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    ConvModule(in_channels=cls_out_channels,
                               out_channels=cls_out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               norm_cfg=self.norm_cfg,
                               act_cfg=self.act_cfg),
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=(self.num_base_priors + self.reg_max) * 1,
                              kernel_size=1)))
            if self.use_bn_head:
                if self.reparameterized:
                    self.cls_contrasts.append(
                        RepBNContrastiveHead(self.embed_dims, self.num_guide,
                                      self.norm_cfg))
                else:
                    self.cls_contrasts.append(
                        BNContrastiveHead(self.embed_dims,
                                      self.norm_cfg)
                    )
            else:
                if self.reparameterized:
                    self.cls_contrasts.append(
                        RepBNContrastiveHead(self.embed_dims, self.num_guide,
                                      self.norm_cfg))
                else:
                    self.cls_contrasts.append(
                        ContrastiveHead(self.embed_dims))

        if self.reg_max > 1:
            proj = torch.arange(
                self.reg_max + self.num_base_priors, dtype=torch.float)
            self.register_buffer('proj', proj, persistent=False)

        if self.freeze_all:
            self._freeze_all()

    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False
    
    def forward_train(self,
                      img_feats,
                      txt_feats,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(img_feats, txt_feats)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def forward(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions.
        """
        assert len(img_feats) == self.num_levels
        if txt_feats is not None:
            txt_feats = txt_feats.permute(1,0,2) # k, b, c -> b, k, c
            txt_feats = tuple(txt_feats for _ in range(len(img_feats)))
            return multi_apply(self.forward_single, img_feats, txt_feats, 
                                self.cls_preds, self.reg_preds, self.ang_preds,
                                self.cls_contrasts, self.scales, self.featmap_strides)
        else:
            return multi_apply(self.forward_single_wot, img_feats, 
                                self.cls_preds, self.reg_preds, self.ang_preds,
                                self.cls_contrasts, self.scales, self.featmap_strides)
            
    def forward_single_wot(self, img_feat: Tensor, 
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList, ang_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList,scale: List, stride) -> Tuple[Tensor, Tensor]:
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed)
        
        bbox_dist_preds = reg_pred(img_feat)
        predicted_angle = ang_pred(img_feat)
        
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max + self.num_base_priors,
                 h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = scale(bbox_dist_preds).float()
            bbox_preds = bbox_preds.clamp(min=0)
            if not self.training:
                bbox_preds *= stride
        if self.training:
            return cls_logit , bbox_preds, predicted_angle
        else:
            return cls_logit , bbox_preds, predicted_angle
    
    def forward_single(self, img_feat: Tensor, txt_feat: Tensor ,
                       cls_pred: nn.ModuleList, reg_pred: nn.ModuleList, ang_pred: nn.ModuleList,
                       cls_contrast: nn.ModuleList,scale: List, stride) -> Tuple[Tensor, Tensor]:
        b, _, h, w = img_feat.shape
        cls_embed = cls_pred(img_feat)
        cls_logit = cls_contrast(cls_embed, txt_feat)
        
        bbox_dist_preds = reg_pred(img_feat)
        predicted_angle = ang_pred(img_feat)
        
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape(
                [-1, 4, self.reg_max + self.num_base_priors,
                 h * w]).permute(0, 3, 1, 2)

            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(
                self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = scale(bbox_dist_preds).float()
            bbox_preds = bbox_preds.clamp(min=0)
            if not self.training:
                bbox_preds *= stride
        if self.training:
            return cls_logit , bbox_preds, predicted_angle
        else:
            return cls_logit , bbox_preds, predicted_angle
        
    @force_fp32(
    apply_to=('cls_scores', 'bbox_preds', 'angle_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   angle_preds,
                   img_metas,
                   cfg=None,
                   rescale=None):
        
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas['filename'])):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            angle_pred_list = [
                angle_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas['img_shape'][img_id]
            scale_factor = img_metas['scale_factor'][img_id]
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 angle_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list
        
    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           angle_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, angle_pred, points in zip(
                cls_scores, bbox_preds, angle_preds, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            score = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            if self.contrast_softmax:
                scores = score.softmax(-1)
            else:
                scores = score.sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes) 
        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img)
        return det_bboxes, det_labels