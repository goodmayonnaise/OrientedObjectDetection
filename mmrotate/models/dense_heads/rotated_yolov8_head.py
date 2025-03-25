# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple, Union
import math
import torch, gc
from torch import Tensor
import torch.nn as nn
from ..builder import ROTATED_HEADS, build_loss
from ..blocks import *
from mmcv.cnn import ConvModule
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
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
class RotatedYOLOv8Head(BaseDenseHead):
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
        super().__init__(init_cfg)

        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.regress_range = regress_ranges
        self.num_levels = len(self.featmap_strides)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.reg_max = reg_max
    
        self.cls_out_channels = num_classes
        
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', True)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        
        self.in_channels = in_channels
        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
            
        self.in_channels = in_channels

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.prior_generator = MlvlPointGenerator(featmap_strides)
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        
        

        
        self._init_layers()
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
        else:
            self.assigner = None
    
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
                
        # max_scores, pred_labels = torch.max(cls_logit.softmax(dim=-1), dim=-1)
        # bg_scores = cls_logit[:, -1].sigmoid().mean().item()  # BG 평균 확률\
        # print(f"Max FG score: {max_scores.mean().item():.4f}, BG score: {bg_scores:.4f}")

        if self.training:
            return cls_logit, bbox_preds, predicted_angle
        else:
            return cls_logit, bbox_preds, predicted_angle
    
    def loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             bbox_dist_preds=None,
             gt_bboxes_ignore=None):

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
        bg_class_ind = 15 # jyjyjyjyjyjy
        # bg_class_ind = self.num_classes
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
            
            if self.loss_cls.__str__() == 'VarifocalLoss()':
                bbox_overlap = build_loss(dict(type='RotatedIoULoss', loss_weight=1.0,
                               reduction='none', mode='linear')) ##
                iou_targets = 1 - bbox_overlap(
                    pos_decoded_bbox_preds,
                    pos_decoded_target_preds.detach()).clamp(min=1e-6)
                # pos_ious = torch.sqrt(iou_targets.clone().detach() * pos_centerness_targets.clone().detach()[:,0]) + 1e-6
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)
                cls_iou_targets[pos_inds, flatten_labels[pos_inds]] = iou_targets
                loss_cls = self.loss_cls(
                    flatten_cls_scores,
                    cls_iou_targets,
                    avg_factor=num_pos)
            elif 'AuxFocalLoss' in self.loss_cls.__str__():
                loss_cls, loss_cls_bg = self.loss_cls(
                    flatten_cls_scores, flatten_labels, avg_factor=num_pos
                )
                return dict(
                    loss_cls=loss_cls,
                    loss_bbox=loss_bbox,
                    loss_cls_bg=loss_cls_bg
                )
            else:     
                loss_cls = self.loss_cls(
                    flatten_cls_scores, flatten_labels, avg_factor=num_pos)
            
            # loss_dfl = self.loss_dfl(
            #     pos_dist_preds.reshape(-1, self.reg_max),
            #     pos_bbox_targets[:,:4].reshape(-1))
        else:
            loss_bbox = pos_bbox_preds.sum()
            # loss_bbox = pos_bbox_preds.sum() * 0 ## 없는게 더 나을듯..
            # loss_dfl = pos_bbox_preds.sum() * 0
        
        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox)
            
        
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
    
@ROTATED_HEADS.register_module()
class RotatedYOLOv8AngleHead(RotatedYOLOv8Head):
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
                 test_cfg=None,
                 loss_angle=None):
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
        self.loss_angle = build_loss(loss_angle)

    def loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
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
        # bg_class_ind = 15 # jyjyjyjyjyjy
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

            loss_cls = self.loss_cls(
                flatten_cls_scores, flatten_labels, avg_factor=num_pos)
            
            loss_angle = self.loss_angle(
                flatten_angle_preds, flatten_angle_targets, avg_factor=num_pos
            )
        else:
            loss_bbox = pos_bbox_preds.sum()
            # loss_bbox = pos_bbox_preds.sum() * 0 ## 없는게 더 나을듯..
            # loss_dfl = pos_bbox_preds.sum() * 0
        
        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_angle=loss_angle)
            