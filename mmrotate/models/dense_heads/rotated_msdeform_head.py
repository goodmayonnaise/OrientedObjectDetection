

# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple, Union
import math 

import torch
from torch import Tensor
import torch.nn as nn

from .rotated_yolov8_head import RotatedYOLOv8Head
from ..builder import ROTATED_HEADS, build_loss
from ..blocks import *

from mmcv.cnn import ConvModule, Scale
from mmcv.ops import DeformConv2dPack
from mmdet.core import multi_apply
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from mmrotate.core import build_bbox_coder

INF = 1e8

@ROTATED_HEADS.register_module()
class RotatedMSDCNHead(RotatedYOLOv8Head):
    """YOLOv8 Head"""
    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 regress_ranges= ((-1, 96), (96, 192), (192, 384)),  ## ((-1, 64), (64, 128), (128, 256)) # add la
                 reg_max = 0,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 kernel_size=[3, 5, 7],
                 bbox_coder=dict(type='DistanceAnglePointCoder', angle_version='le90'),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 matching:int=0,
                 debug=False,
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
        self.kernel_size = kernel_size
        self.num_levels = len(featmap_strides)
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        super().__init__(num_classes, in_channels, widen_factor, reg_max, featmap_strides,
                         regress_ranges=regress_ranges, matching=matching, debug=debug,
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
                5 / self.num_classes / (1024 / stride)**2)
                # 5 / self.num_classes / (640 / stride)**2)

    def _init_layers(self):
        # from mmrotate.models.utils import pack

        self.deform = nn.ModuleList()
        self.deform2 = nn.ModuleList()
        self.deform3 = nn.ModuleList()
        
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.ang_preds = nn.ModuleList()
        
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.featmap_strides])
        reg_out_channels = max(
            (16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(3*self.in_channels[0]//8, self.num_classes)

        for i in range(self.num_levels):
            self.deform.append(nn.Sequential(
                DeformConv2dPack(in_channels=self.in_channels[i],
                                 out_channels=self.in_channels[i]//2,
                                 kernel_size=self.kernel_size[0],
                                 stride=1, padding=1),
                nn.BatchNorm2d(self.in_channels[i]//2, eps=self.norm_cfg['eps'], momentum=self.norm_cfg['momentum']),
                nn.SiLU(self.act_cfg['inplace']),
                
                DeformConv2dPack(in_channels=self.in_channels[i]//2,
                                 out_channels=self.in_channels[i]//4,
                                 kernel_size=self.kernel_size[0],
                                 stride=1, padding=1),
                nn.BatchNorm2d(self.in_channels[i]//4, eps=self.norm_cfg['eps'], momentum=self.norm_cfg['momentum']),
                nn.SiLU(self.act_cfg['inplace']),
                
                DeformConv2dPack(in_channels=self.in_channels[i]//4,
                                 out_channels=self.in_channels[i]//8,
                                 kernel_size=self.kernel_size[0],
                                 stride=1, padding=1),
                nn.BatchNorm2d(self.in_channels[i]//8, eps=self.norm_cfg['eps'], momentum=self.norm_cfg['momentum']),
                nn.SiLU(self.act_cfg['inplace']),
                                 ))
            self.deform2.append(nn.Sequential(
                DeformConv2dPack(in_channels=self.in_channels[i],
                                 out_channels=self.in_channels[i]//2,
                                 kernel_size=self.kernel_size[1],
                                 stride=1, padding=2),
                nn.BatchNorm2d(self.in_channels[i]//2, eps=self.norm_cfg['eps'], momentum=self.norm_cfg['momentum']),
                nn.SiLU(self.act_cfg['inplace']),
                
                DeformConv2dPack(in_channels=self.in_channels[i]//2,
                                 out_channels=self.in_channels[i]//4,
                                 kernel_size=self.kernel_size[1],
                                 stride=1, padding=2),
                nn.BatchNorm2d(self.in_channels[i]//4, eps=self.norm_cfg['eps'], momentum=self.norm_cfg['momentum']),
                nn.SiLU(self.act_cfg['inplace']),
                
                DeformConv2dPack(in_channels=self.in_channels[i]//4,
                                 out_channels=self.in_channels[i]//8,
                                 kernel_size=self.kernel_size[1],
                                 stride=1, padding=2),
                nn.BatchNorm2d(self.in_channels[i]//8, eps=self.norm_cfg['eps'], momentum=self.norm_cfg['momentum']),
                nn.SiLU(self.act_cfg['inplace'])))

            self.deform3.append(nn.Sequential(
                DeformConv2dPack(in_channels=self.in_channels[i],
                                 out_channels=self.in_channels[i]//2,
                                 kernel_size=self.kernel_size[2],
                                 stride=1, padding=3),
                nn.BatchNorm2d(self.in_channels[i]//2, eps=self.norm_cfg['eps'], momentum=self.norm_cfg['momentum']),
                nn.SiLU(self.act_cfg['inplace']),
                
                DeformConv2dPack(in_channels=self.in_channels[i]//2,
                                 out_channels=self.in_channels[i]//4,
                                 kernel_size=self.kernel_size[2],
                                 stride=1, padding=3),
                nn.BatchNorm2d(self.in_channels[i]//4, eps=self.norm_cfg['eps'], momentum=self.norm_cfg['momentum']),
                nn.SiLU(self.act_cfg['inplace']),
                
                DeformConv2dPack(in_channels=self.in_channels[i]//4,
                                 out_channels=self.in_channels[i]//8,
                                 kernel_size=self.kernel_size[2],
                                 stride=1, padding=3),
                nn.BatchNorm2d(self.in_channels[i]//8, eps=self.norm_cfg['eps'], momentum=self.norm_cfg['momentum']),
                nn.SiLU(self.act_cfg['inplace'])
                ))
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(in_channels=3*self.in_channels[i]//8,
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
                    ConvModule(in_channels=3*self.in_channels[i]//8,
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
                    ConvModule(in_channels=3*self.in_channels[i]//8,
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
        return multi_apply(self.forward_single, x, self.deform, self.deform2, self.deform3,
                            self.cls_preds, self.reg_preds, self.ang_preds, self.scales, self.featmap_strides)

    # def forward_single(self, x: Tensor, deform_conv:nn.Module, deform2_conv:nn.Module, deform3_conv:nn.Module,
    def forward_single(self, x: Tensor, deform_conv:nn.Module, deform2_conv, deform3_conv,
                       cls_pred: nn.Module, reg_pred: nn.Module,  ang_pred: nn.Module, 
                       scale: List, stride) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        
        deform1 = deform_conv(x)
        deform2 = deform2_conv(x)
        deform3 = deform3_conv(x)

        ms_deform = torch.cat([deform1, deform2, deform3], 1)

        cls_logit = cls_pred(ms_deform) # cls scroe
        bbox_dist_preds = reg_pred(ms_deform)
        predicted_angle = ang_pred(ms_deform)

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
