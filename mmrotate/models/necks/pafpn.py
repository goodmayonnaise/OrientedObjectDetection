from ..builder import ROTATED_NECKS
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


def make_round(x: float, deepen_factor: float = 1.0) -> int:
    """Make sure that x*deepen_factor becomes an integer not less than 1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x


def make_divisible(x: float,
                   widen_factor: float = 1.0,
                   divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor

@ROTATED_NECKS.register_module()
class YOLOv6RepPAFPN(BaseYOLONeck):
    """Path Aggregation Network used in YOLOv6.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        freeze_all(bool): Whether to freeze the model.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='ReLU', inplace=True).
        block_cfg (dict): Config dict for the block used to build each
            layer. Defaults to dict(type='RepVGGBlock').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 12,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 block_cfg: ConfigType = dict(type='RepVGGBlock'),
                 init_cfg: OptMultiConfig = None):
        self.num_csp_blocks = num_csp_blocks
        self.block_cfg = block_cfg
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The reduce layer.
        """
        if idx == 2:
            layer = ConvModule(
                in_channels=int(self.in_channels[idx] * self.widen_factor),
                out_channels=int(self.out_channels[idx - 1] *
                                 self.widen_factor),
                kernel_size=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        else:
            layer = nn.Identity()

        return layer

    def build_upsample_layer(self, idx: int) -> nn.Module:
        """build upsample layer.

        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The upsample layer.
        """
        return nn.ConvTranspose2d(
            in_channels=int(self.out_channels[idx - 1] * self.widen_factor),
            out_channels=int(self.out_channels[idx - 1] * self.widen_factor),
            kernel_size=2,
            stride=2,
            bias=True)

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The top down layer.
        """
        block_cfg = self.block_cfg.copy()

        layer0 = RepStageBlock(
            in_channels=int(
                (self.out_channels[idx - 1] + self.in_channels[idx - 1]) *
                self.widen_factor),
            out_channels=int(self.out_channels[idx - 1] * self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            block_cfg=block_cfg)

        if idx == 1:
            return layer0
        elif idx == 2:
            layer1 = ConvModule(
                in_channels=int(self.out_channels[idx - 1] *
                                self.widen_factor),
                out_channels=int(self.out_channels[idx - 2] *
                                 self.widen_factor),
                kernel_size=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            return nn.Sequential(layer0, layer1)

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The downsample layer.
        """
        return ConvModule(
            in_channels=int(self.out_channels[idx] * self.widen_factor),
            out_channels=int(self.out_channels[idx] * self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=3 // 2,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.
        Returns:
            nn.Module: The bottom up layer.
        """
        block_cfg = self.block_cfg.copy()

        return RepStageBlock(
            in_channels=int(self.out_channels[idx] * 2 * self.widen_factor),
            out_channels=int(self.out_channels[idx + 1] * self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            block_cfg=block_cfg)

    def build_out_layer(self, *args, **kwargs) -> nn.Module:
        """build out layer."""
        return nn.Identity()

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

@ROTATED_NECKS.register_module()
class YOLOv8PAFPN(YOLOv6RepPAFPN):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
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

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
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
        return CSPLayerWithTwoConv(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
            add_identity=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
    


@ROTATED_NECKS.register_module()
class YOLOv8PAFPN_E(YOLOv8PAFPN):
    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None,
                 expanded_up_feat_channels=None,
                 expanded_down_feat_channels=None
                 ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            num_csp_blocks=num_csp_blocks,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        self.expanded_down_feat_channels = expanded_down_feat_channels
        self.expanded_up_feat_channels = expanded_up_feat_channels
        
        if self.expanded_down_feat_channels is not None:
            self.build_expanded_down_layers()

        if self.expanded_up_feat_channels is not None:
            self.build_expanded_up_layers()
                
    def build_expanded_down_layers(self):
        self.expanded_down_feat_channels.insert(0, self.out_channels[-1])
        for i in range(len(self.expanded_down_feat_channels)-1):    
            self.downsample_layers.append(ConvModule(
                make_divisible(self.expanded_down_feat_channels[i], 
                            self.widen_factor),
                make_divisible(self.expanded_down_feat_channels[i], 
                            self.widen_factor),
                kernel_size=3, 
                stride=2, 
                padding=1, 
                norm_cfg=self.norm_cfg, 
                act_cfg=self.act_cfg))

        for i in range(len(self.expanded_down_feat_channels)-1):
            self.bottom_up_layers.append(
                CSPLayerWithTwoConv(
                make_divisible(
                    self.expanded_down_feat_channels[i],
                    self.widen_factor),
                make_divisible(
                    self.expanded_down_feat_channels[i+1], 
                    self.widen_factor),
                num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            )

        for idx in range(len(self.expanded_down_feat_channels)-1):
            self.out_layers.append(self.build_out_layer(idx))        

    def build_expanded_up_layers(self):
        self.expanded_up_feat_channels.append(self.out_channels[0])
        self.expanded_up_layers = nn.ModuleList()
        for i in range(1, len(self.expanded_up_feat_channels)):
            self.expanded_up_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                CSPLayerWithTwoConv(
                make_divisible(self.expanded_up_feat_channels[-i], self.widen_factor),
                make_divisible(self.expanded_up_feat_channels[-i-1], self.widen_factor),
                num_blocks=make_round(self.num_csp_blocks, self.deepen_factor),
                add_identity=False,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            ))
        
        for idx in range(len(self.expanded_up_feat_channels)-1):
            self.out_layers.append(self.build_out_layer(idx))     

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

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        return CSPLayerWithTwoConv(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
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
        return CSPLayerWithTwoConv(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
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
            upsample_feat = self.upsample_layers[len(self.in_channels) -1 -idx](
                                                     feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        if self.expanded_up_feat_channels is not None:
            feat = inner_out
            expanded_inner_outs = []
            for i in range(len(self.expanded_up_feat_channels)-1):
                out_feat = self.expanded_up_layers[-i](feat)
                if len(expanded_inner_outs)==0:
                    expanded_inner_outs.append(out_feat)    
                else:
                    expanded_inner_outs.insert(0, out_feat)
                feat = out_feat
        
        # bottom-up path
        outs = [inner_outs[0]]                
        for idx in range(len(self.in_channels)-1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)
        
        # expanded bottom-up path 
        if self.expanded_down_feat_channels is not None:
            feat_low = outs[-1]
            for idx in range(len(self.expanded_down_feat_channels)-1, 0, -1):
                downsample_feat = self.downsample_layers[-idx](feat_low)
                out = self.bottom_up_layers[-idx](downsample_feat)
                outs.append(out)
                feat_low = out 
        if self.expanded_up_feat_channels is not None:
            outs = expanded_inner_outs + outs
        
        # out_layers
        results = []
        for idx in range(len(outs)):
            results.append(self.out_layers[idx](outs[idx]))

        return results
