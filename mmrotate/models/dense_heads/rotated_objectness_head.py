

# Copyright (c) OpenMMLab. All rights reserved.
import math 
from typing import  Sequence, Union, Tuple, List

import torch
from torch import Tensor, nn

from ..blocks import *
from ..builder import ROTATED_HEADS, build_loss
from .rotated_yolov8_head import RotatedYOLOv8Head


from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32
from mmdet.core import reduce_mean, multi_apply
from mmrotate.core import multiclass_nms_rotated



@ROTATED_HEADS.register_module()
class RotatedDecoupledBGHead(RotatedYOLOv8Head):
    """YOLOv8 Head"""
    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 reg_max = 0,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 regress_ranges=((-1, 96), (96, 192), (192, 384)),
                 bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bg=dict(
                     type='CustomCrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0
                 ),
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
        
        super().__init__(
            num_classes=num_classes,
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
            test_cfg=test_cfg   
        )
        
    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        # super().init_weights()
        for reg_pred, cls_pred, ang_pred, obj_pred, stride in zip(self.reg_preds, self.cls_preds, self.ang_preds, self.obj_preds,
                                              self.featmap_strides):
            reg_pred[-1].bias.data[:] = 1.0  # box
            ang_pred[-1].bias.data[:] = 1.0  # angle
            # cls (.01 objects, 80 classes, 640 img)
            cls_pred[-1].bias.data[:self.num_classes] = math.log(
                5 / self.num_classes / (1024 / stride)**2)
            obj_pred[-1].bias.data[:] = 1.0
            
    def _init_layers(self):
        
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.ang_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.featmap_strides])
        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.obj_preds.append(
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
                              out_channels=1,
                              kernel_size=1))
                )
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

        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, 
                           self.cls_preds, self.reg_preds, self.ang_preds, self.obj_preds,
                           self.scales, self.featmap_strides)

    def forward_single(self, x: Tensor, cls_pred: nn.Module, reg_pred: nn.Module,  
                       ang_pred: nn.Module, obj_pred: nn.Module,
                       scale: List, stride) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        obj_logit = obj_pred(x)
        obj_logit = torch.sigmoid(obj_logit)
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
        if self.training :
            return cls_logit, bbox_preds, predicted_angle, obj_logit
        else:
            return cls_logit, bbox_preds, predicted_angle, obj_logit

    def loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
             obj_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             bbox_dist_preds = None,
             gt_bboxes_ignore = None):

        assert len(cls_scores) == len(bbox_preds) == len(angle_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        self.img_metas = img_metas[0]

        labels, bbox_targets, angle_targets = self.assigner(
            all_level_points, gt_bboxes, gt_labels, 
            bbox_preds, angle_preds, cls_scores, self.img_metas)

        num_imgs = cls_scores[0].size(0)
        flatten_obj_scores = [
            obj_pred.permute(0, 2, 3, 1).reshape(-1, 1) 
            for obj_pred in obj_preds
        ]
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
        # flatten_dist_preds = [
        #     bbox_pred_org.reshape(num_imgs, -1, self.reg_max * 4).reshape(-1, self.reg_max*4)
        #     for bbox_pred_org in bbox_dist_preds
        # ]
        
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_obj_scores = torch.cat(flatten_obj_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        # flatten_dist_preds = torch.cat(flatten_dist_preds)
        
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
        # pos_dist_preds = flatten_dist_preds[pos_inds]

        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        
        # if (pos_bbox_targets > 16).any().item() :
            # print(pos_bbox_targets.shape)
            # print(pos_bbox_targets.argmax(), pos_bbox_targets.max())
        
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

            loss_cls = self.loss_cls(flatten_cls_scores, flatten_obj_scores, flatten_labels, avg_factor=num_pos)
 
        else:
            loss_bbox = pos_bbox_preds.sum()
            # loss_bbox = pos_bbox_preds.sum() * 0 ## 없는게 더 나을듯..
            # loss_dfl = pos_bbox_preds.sum() * 0
         
        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox)
        

    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'obj_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   angle_preds,
                   obj_preds,
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

        
'''
동일 cls head에서 1x1만 decouple obj, cls  
'''
@ROTATED_HEADS.register_module()
class RotatedDecoupled1x1ObjHead(RotatedDecoupledBGHead):
    """YOLOv8 Head"""
    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 reg_max = 0,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 regress_ranges=((-1, 96), (96, 192), (192, 384)),
                 bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bg=dict(
                     type='CustomCrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0
                 ),
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
        
        super().__init__(
            num_classes=num_classes,
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
            test_cfg=test_cfg   
        )
        
    def init_weights(self, prior_prob=0.01):
        for reg_pred, cls_pred, ang_pred, obj_pred, stride in zip(self.reg_preds, self.fg_preds, self.ang_preds, self.obj_preds,
                                              self.featmap_strides):
            reg_pred[-1].bias.data[:] = 1.0  # box
            ang_pred[-1].bias.data[:] = 1.0  # angle
            # cls (.01 objects, 80 classes, 640 img)
            cls_pred[-1].bias.data[:self.num_classes] = math.log(
                5 / self.num_classes / (1024 / stride)**2)
            obj_pred[-1].bias.data[:] = 1.0
            
    def _init_layers(self):
        
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.ang_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.fg_preds = nn.ModuleList()
        
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.featmap_strides])
        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.obj_preds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=1,
                              kernel_size=1))
                )
            self.fg_preds.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=cls_out_channels,
                              out_channels=self.num_classes,
                              kernel_size=1)
                )
            )
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
                               act_cfg=self.act_cfg)))

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

        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, 
                           self.cls_preds, self.reg_preds, self.ang_preds, self.obj_preds, self.fg_preds,
                           self.scales, self.featmap_strides)

    def forward_single(self, x: Tensor, cls_pred: nn.Module, reg_pred: nn.Module,  
                       ang_pred: nn.Module, obj_pred: nn.Module, fg_pred: nn.Module,
                       scale: List, stride) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        cls_logit = cls_pred(x)
        obj_logit = torch.sigmoid(obj_pred(cls_logit))
        fg_logit = fg_pred(cls_logit)

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
        return fg_logit, bbox_preds, predicted_angle, obj_logit
