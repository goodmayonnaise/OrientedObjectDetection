# Copyright (c) OpenMMLab. All rights reserved.
from typing import List
from typing import List, Optional, Sequence, Tuple, Union
import math 
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
# from mmdet.utils import ConfigType, OptMultiConfig
from mmdet.models.builder import MODELS
from torch import Tensor
from .base_yolo_neck import BaseYOLONeck
from ..blocks import *
from .pafpn import YOLOv8PAFPN

def make_divisible(x: float,
                   widen_factor: float = 1.0,
                   divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor

def make_round(x: float, deepen_factor: float = 1.0) -> int:
    """Make sure that x*deepen_factor becomes an integer not less than 1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x

@MODELS.register_module()
class AttentionalYOLOv8PAFPN(YOLOv8PAFPN):
    def __init__(self,
                 lska_reduce: bool = True,
                 **kwargs):
        self.lska_reduce = lska_reduce
        super().__init__(**kwargs)
        
    def build_reduce_layer(self, idx: int) -> nn.Module:
        if self.lska_reduce:
            return LSKA(
                in_channels = make_divisible(self.in_channels[idx], self.widen_factor),
                norm_cfg = self.norm_cfg
            ) 
        else:
            return nn.Identity()
            
        
    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """           
        
        return C2fCBAM(
            in_channels=make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            out_channels=make_divisible(self.out_channels[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return C2fCBAM(
            in_channels=make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            out_channels=make_divisible(self.out_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)


@MODELS.register_module()
class AttentionalYOLOv8PAFPN2(YOLOv8PAFPN):
    def __init__(self,
                 lska_reduce: bool = False,
                 with_cbam: bool = False,
                 **kwargs):
        self.lska_reduce = lska_reduce
        self.with_cbam = with_cbam
        super().__init__(**kwargs)
        
        self.asff_down_layers = nn.ModuleList()
        for idx in range(len(self.in_channels) - 1, 0, -1):
            self.asff_down_layers.append(self.build_asff_layer(idx, down=True))
            
        
        self.asff_up_layers = nn.ModuleList()
        for idx in range(len(self.in_channels) - 1):
            self.asff_up_layers.append(self.build_asff_layer(idx, down=False))
            
        
        
    def build_asff_layer(self, idx: int, down: bool):
        if down:
            return ASFFDown(
                in_channels = make_divisible(self.in_channels[idx], self.widen_factor),
                out_channels = make_divisible(self.in_channels[idx-1], self.widen_factor),
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
        else:
            return ASFFDown(
                in_channels = make_divisible(self.in_channels[idx], self.widen_factor),
                out_channels = make_divisible(self.in_channels[idx+1], self.widen_factor),
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )
            
        
    def build_reduce_layer(self, idx: int) -> nn.Module:
        if self.lska_reduce:
            return LSKA(
                in_channels = make_divisible(self.in_channels[idx], self.widen_factor),
                norm_cfg = self.norm_cfg
            ) 
        else:
            return nn.Identity()
            
    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        in_channels = out_channels = make_divisible(self.in_channels[idx - 1], self.widen_factor)
        if self.with_cbam:
            return C2fCBAM(
                in_channels=in_channels,
                out_channels=out_channels,
                num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            return CSPLayerWithTwoConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                    add_identity=False,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        in_channels = out_channels = make_divisible(self.out_channels[idx + 1], self.widen_factor)

        if self.with_cbam:
            return C2fCBAM(
                in_channels=in_channels,
                out_channels=out_channels,
                num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            return CSPLayerWithTwoConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                    add_identity=False,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
        
    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                                                     feat_high)
            top_down_layer_inputs = self.asff_down_layers[len(self.in_channels) - 1 -
                                                 idx](feat_low, upsample_feat)
            # if self.upsample_feats_cat_first:
            #     top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            # else:
            #     top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            bottom_up_layer_inputs = self.asff_up_layers[idx](feat_high, downsample_feat)
            out = self.bottom_up_layers[idx](
                bottom_up_layer_inputs)
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)
    
@MODELS.register_module()
class AttentionalYOLOv8PAFPN3(YOLOv8PAFPN):
    def __init__(self,
                 lska_reduce: bool = False,
                 with_cbam: bool = False,
                 cross_attention = False,
                 **kwargs):
        self.cross_attention = cross_attention
        super().__init__(**kwargs) 
        if self.cross_attention :
            self.asff_down_layer = DCASFF(
                in_channels = make_divisible(self.in_channels[1], self.widen_factor),
                out_channels = make_divisible(self.in_channels[0], self.widen_factor),
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            self.asff_down_layer = DASFF(
                in_channels = make_divisible(self.in_channels[1], self.widen_factor),
                out_channels = make_divisible(self.in_channels[0], self.widen_factor),
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        # for idx in range(len(self.in_channels) - 1, 0, -1):
        #     self.asff_down_layers.append(self.build_asff_layer(idx, down=True))
            
        # self.asff_up_layers = nn.ModuleList()
        # for idx in range(len(self.in_channels) - 1):
        #     self.asff_up_layers.append(self.build_asff_layer(idx, down=False))
              
    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        if idx == 1:
            in_channels = out_channels = make_divisible(self.in_channels[idx - 1], self.widen_factor)
        # if self.with_cbam:
        #     return C2fCBAM(
        #         in_channels=in_channels,
        #         out_channels=out_channels,
        #         num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
        #         add_identity=False,
        #         norm_cfg=self.norm_cfg,
        #         act_cfg=self.act_cfg)
        # else:
            return CSPLayerWithTwoConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                    add_identity=False,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
        else:
            return CSPLayerWithTwoConv(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    # def build_bottom_up_layer(self, idx: int) -> nn.Module:
    #     """build bottom up layer.

    #     Args:
    #         idx (int): layer idx.

    #     Returns:
    #         nn.Module: The bottom up layer.
    #     """
    #     in_channels = out_channels = make_divisible(self.out_channels[idx + 1], self.widen_factor)

    #     if self.with_cbam:
    #         return C2fCBAM(
    #             in_channels=in_channels,
    #             out_channels=out_channels,
    #             num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
    #             add_identity=False,
    #             norm_cfg=self.norm_cfg,
    #             act_cfg=self.act_cfg)
    #     else:
    #         return CSPLayerWithTwoConv(
    #                 in_channels=in_channels,
    #                 out_channels=out_channels,
    #                 num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
    #                 add_identity=False,
    #                 norm_cfg=self.norm_cfg,
    #                 act_cfg=self.act_cfg)
            
    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                                                     feat_high)
            if idx == 1 :
                top_down_layer_inputs = self.asff_down_layer(feat_low, upsample_feat)
            else:    
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            # bottom_up_layer_inputs = self.asff_up_layers[idx](feat_high, downsample_feat)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)
     