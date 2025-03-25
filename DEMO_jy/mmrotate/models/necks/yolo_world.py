# Copyright (c) Tencent Inc. All rights reserved.
import copy
from typing import List, Union
import math
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Linear

from torch import Tensor
# from mmdet.utils import ConfigType, OptMultiConfig
from ..builder import ROTATED_NECKS, build_neck
from mmdet.models.builder import MODELS
from .base_yolo_neck import BaseYOLONeck
from ..blocks import *

# from mmyolo.registry import MODELS
# from mmyolo.models.utils import make_divisible, make_round
# from mmyolo.models.necks.yolov8_pafpn import YOLOv8PAFPN

def make_divisible(x: float,
                   widen_factor: float = 1.0,
                   divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor

def make_round(x: float, deepen_factor: float = 1.0) -> int:
    """Make sure that x*deepen_factor becomes an integer not less than 1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x


@ROTATED_NECKS.register_module()
class YOLOWorldPAFPN(BaseYOLONeck):
    """Path Aggregation Network used in YOLO World
    Following YOLOv8 PAFPN, including text to image fusion
    """
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 guide_channels: int,
                 embed_channels: List[int],
                 num_heads: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 block_cfg = dict(type='CSPLayerWithTwoConv'),
                 norm_cfg = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 act_cfg = dict(type='SiLU', inplace=True),
                 init_cfg = None) -> None:
        self.guide_channels = guide_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.block_cfg = block_cfg
        self.num_csp_blocks=num_csp_blocks
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         deepen_factor=deepen_factor,
                         widen_factor=widen_factor,
                         freeze_all=freeze_all,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)
        
    def init_weights(self):
        if self.init_cfg is None:
            """Initialize the parameters."""
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    # In order to be consistent with the source code,
                    # reset the Conv2d initialization parameters
                    m.reset_parameters()
        else:
            super().init_weights()
    
    def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
        """build upsample layer."""
        return nn.Upsample(scale_factor=2, mode='nearest')
    
    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        return ConvModule(
            make_divisible(self.in_channels[idx], self.widen_factor),
            make_divisible(self.in_channels[idx], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
    
    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The reduce layer.
        """
        return nn.Identity()

    def build_out_layer(self, *args, **kwargs) -> nn.Module:
        """build out layer."""
        return nn.Identity()
    
    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        block_cfg = copy.deepcopy(self.block_cfg)
        
        block_cfg.update(
            dict(in_channels=make_divisible(
                (self.in_channels[idx - 1] + self.in_channels[idx]),
                self.widen_factor),
                 out_channels=make_divisible(self.out_channels[idx - 1],
                                             self.widen_factor),
                 guide_channels=self.guide_channels,
                 embed_channels=make_round(self.embed_channels[idx - 1],
                                           self.widen_factor),
                 num_heads=make_round(self.num_heads[idx - 1],
                                      self.widen_factor),
                 num_blocks=make_round(self.num_csp_blocks,
                                       self.deepen_factor),
                 add_identity=False,
                 norm_cfg=self.norm_cfg,
                 act_cfg=self.act_cfg))
        return MODELS.build(block_cfg)
    
    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        block_cfg = copy.deepcopy(self.block_cfg)
        block_cfg.update(
            dict(in_channels=make_divisible(
                self.out_channels[idx] + self.out_channels[idx + 1],
                self.widen_factor),
                 out_channels=make_divisible(self.out_channels[idx + 1],
                                             self.widen_factor),
                 guide_channels=self.guide_channels,
                 embed_channels=make_round(self.embed_channels[idx + 1],
                                           self.widen_factor),
                 num_heads=make_round(self.num_heads[idx + 1],
                                      self.widen_factor),
                 num_blocks=make_round(self.num_csp_blocks,
                                       self.deepen_factor),
                 add_identity=False,
                 norm_cfg=self.norm_cfg,
                 act_cfg=self.act_cfg))
        return MODELS.build(block_cfg)

    def forward(self, img_feats: List[Tensor], txt_feats: Tensor = None) -> tuple:
        """Forward function.
        including multi-level image features, text features: BxLxD
        """
        assert len(img_feats) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs, txt_feats)
            inner_outs.insert(0, inner_out)
            
        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat(
                [downsample_feat, feat_high], 1), txt_feats)
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)


@MODELS.register_module()
class YOLOWorldDualPAFPN(YOLOWorldPAFPN):
    """Path Aggregation Network used in YOLO World v8."""
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 guide_channels: int,
                 embed_channels: List[int],
                 num_heads: List[int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 after_bup: bool = False,
                 withbup: bool = True,
                 text_enhancer = dict(
                     type='ImagePoolingAttentionModule',
                     embed_channels=256,
                     num_heads=8,
                     pool_size=3,
                     ),
                 bic : bool = False,
                 text_enhancer_depth : int = 3,
                 block_cfg = dict(type='CSPLayerWithTwoConv'),
                 norm_cfg = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 act_cfg = dict(type='SiLU', inplace=True),
                 init_cfg = None) -> None:
        self.extra_in_channel = in_channels[0]
        self.bic = bic
        self.withbup = withbup
        self.after_bup = after_bup
        if len(in_channels) != len(out_channels):
            in_c = in_channels[1:]
        else:
            in_c = in_channels
        super().__init__(in_channels=in_c,
                         out_channels=out_channels,
                         guide_channels=guide_channels,
                         embed_channels=embed_channels,
                         num_heads=num_heads,
                         deepen_factor=deepen_factor,
                         widen_factor=widen_factor,
                         num_csp_blocks=num_csp_blocks,
                         freeze_all=freeze_all,
                         block_cfg=block_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)
        text_enhancer.update(
            dict(
                image_channels=[int(x * widen_factor) for x in out_channels],
                text_channels=guide_channels,
                num_feats=len(out_channels),
            ))
        # print(text_enhancder)
        self.text_enhancer = nn.ModuleList()
        for _ in range(text_enhancer_depth):
            self.text_enhancer.append(MODELS.build(text_enhancer))
    
    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The reduce layer.
        """
        
        return nn.Identity()
            
    def build_upsample_layer(self, idx: int, *args, **kwargs) -> nn.Module:
        """build upsample layer."""
        
        if self.bic: # YOLOv6_v3.0
            in_channels1 = self.in_channels[
                idx - 2] if idx > 1 else self.extra_in_channel
            return BiFusion(
                in_channels0=int(self.in_channels[idx - 1] * self.widen_factor),
                in_channels1=int(in_channels1 * self.widen_factor),
                out_channels=int(self.out_channels[idx - 1] * self.widen_factor),
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            return nn.Upsample(scale_factor=2, mode='nearest')
        
    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        block_cfg = copy.deepcopy(self.block_cfg)
        
        block_cfg.update(
            dict(in_channels=make_divisible(
                (self.in_channels[idx - 1] + self.in_channels[idx]),
                self.widen_factor),
                 out_channels=make_divisible(self.out_channels[idx - 1],
                                             self.widen_factor),
                 guide_channels=self.guide_channels,
                 embed_channels=make_round(self.embed_channels[idx - 1],
                                           self.widen_factor),
                 num_heads=make_round(self.num_heads[idx - 1],
                                      self.widen_factor),
                 num_blocks=make_round(self.num_csp_blocks,
                                       self.deepen_factor),
                 add_identity=False,
                 norm_cfg=self.norm_cfg,
                 act_cfg=self.act_cfg))
        return MODELS.build(block_cfg)
        
    def forward(self, img_feats: List[Tensor], txt_feats: Tensor) -> tuple:
        """Forward function."""
        # assert len(img_feats) == len(self.in_channels)
        if self.bic:
            # reduce layers
            reduce_outs = [img_feats[0]]
            for idx in range(len(self.in_channels)):
                reduce_outs.append(self.reduce_layers[idx](img_feats[idx + 1]))

            # top-down path
            inner_outs = [reduce_outs[-1]]
            for idx in range(len(self.in_channels) - 1, 0, -1):
                feat_high = inner_outs[0]
                feat_cur = reduce_outs[idx]
                feat_low = reduce_outs[idx - 1]
                # upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)
                top_down_layer_inputs = self.upsample_layers[len(self.in_channels) - 1 -
                                                    idx]([feat_high, feat_cur, feat_low], idx)
                # if self.upsample_feats_cat_first:
                #     top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
                # else:
                #     top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
                inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                    top_down_layer_inputs, txt_feats)
                inner_outs.insert(0, inner_out)

            for ipooling in self.text_enhancer:
                txt_feats = ipooling(txt_feats, inner_outs) 
                
            # bottom-up path
            txt_feats = txt_feats.permute(1,0,2)
            outs = [inner_outs[0]]
            for idx in range(len(self.in_channels) - 1):
                feat_low = outs[-1]
                feat_high = inner_outs[idx + 1]
                downsample_feat = self.downsample_layers[idx](feat_low)
                out = self.bottom_up_layers[idx](torch.cat(
                    [downsample_feat, feat_high], 1), txt_feats)
                outs.append(out)

            # out_layers
            results = []
            for idx in range(len(self.in_channels)):
                results.append(self.out_layers[idx](outs[idx]))

            return tuple(results)
        else:
            # reduce layers
            reduce_outs = []
            for idx in range(len(self.in_channels)):
                reduce_outs.append(self.reduce_layers[idx](img_feats[idx]))
            # top-down path
            inner_outs = [reduce_outs[-1]]
            for idx in range(len(self.in_channels) - 1, 0, -1):
                feat_high = inner_outs[0]
                feat_low = reduce_outs[idx - 1]
                upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](feat_high)
                if self.upsample_feats_cat_first:
                    top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
                else:
                    top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
                inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                    top_down_layer_inputs, txt_feats)
                inner_outs.insert(0, inner_out)
            
            for ipooling in self.text_enhancer:
                txt_feats = ipooling(txt_feats, inner_outs) 
                
            # bottom-up path
            if self.withbup:
                outs = [inner_outs[0]]
                for idx in range(len(self.in_channels) - 1):
                    feat_low = outs[-1]
                    feat_high = inner_outs[idx + 1]
                    downsample_feat = self.downsample_layers[idx](feat_low)
                    out = self.bottom_up_layers[idx](torch.cat(
                        [downsample_feat, feat_high], 1), txt_feats)
                    outs.append(out)
            else:
                outs = inner_outs
            
            if self.after_bup:
                for ipooling in self.text_enhancer:
                    txt_feats = ipooling(txt_feats, inner_outs) 

            txt_feats = txt_feats.permute(1,0,2)
            results = []
            
            # out_layers
            for idx in range(len(self.in_channels)):
                results.append(self.out_layers[idx](outs[idx]))

            return tuple(results)