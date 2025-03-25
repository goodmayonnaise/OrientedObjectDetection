# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Union
from torch import Tensor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

import einops
# from mmdet.utils import ConfigType, OptMultiConfig
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm
from typing import List, Optional, Sequence, Tuple, Union
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule, MaxPool2d, Linear,
                      build_norm_layer)
from mmengine.config import ConfigDict
from mmdet.models.builder import MODELS
# from mmdet.models.layers.csp_layer import DarknetBottleneck as MMDET_DarknetBottleneck

ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
# Type hint of one or more config data
MultiConfig = Union[ConfigType, List[ConfigType]]
OptMultiConfig = Optional[MultiConfig]

class SiLU(nn.Module):
    """Sigmoid Weighted Liner Unit."""

    def __init__(self, inplace=True):
        super().__init__()

    def forward(self, inputs) -> Tensor:
        return inputs * torch.sigmoid(inputs)
MODELS.register_module(module=SiLU, name='SiLU')
MODELS.register_module(module=nn.ReLU, name='ReLU')

@MODELS.register_module()
class RepVGGBlock(nn.Module):
    """RepVGGBlock is a basic rep-style block, including training and deploy
    status This code is based on
    https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution. Default: 1
        padding (int, tuple): Padding added to all four sides of
            the input. Default: 1
        dilation (int or tuple): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        padding_mode (string, optional): Default: 'zeros'
        use_se (bool): Whether to use se. Default: False
        use_alpha (bool): Whether to use `alpha` parameter at 1x1 conv.
            In PPYOLOE+ model backbone, `use_alpha` will be set to True.
            Default: False.
        use_bn_first (bool): Whether to use bn layer before conv.
            In YOLOv6 and YOLOv7, this will be set to True.
            In PPYOLOE, this will be set to False.
            Default: True.
        deploy (bool): Whether in deploy mode. Default: False
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 1,
                 dilation: Union[int, Tuple[int]] = 1,
                 groups: Optional[int] = 1,
                 padding_mode: Optional[str] = 'zeros',
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 use_se: bool = False,
                 use_alpha: bool = False,
                 use_bn_first=True,
                 deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU(inplace = True)

        if use_se:
            raise NotImplementedError('se block not supported yet')
        else:
            self.se = nn.Identity()

        if use_alpha:
            alpha = torch.ones([
                1,
            ], dtype=torch.float32, requires_grad=True)
            self.alpha = nn.Parameter(alpha, requires_grad=True)
        else:
            self.alpha = None

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode)

        else:
            if use_bn_first and (out_channels == in_channels) and stride == 1:
                self.rbr_identity = build_norm_layer(
                    norm_cfg, num_features=in_channels)[1]
            else:
                self.rbr_identity = None

            self.rbr_dense = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.rbr_1x1 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward process.
        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        if self.alpha:
            return self.nonlinearity(
                self.se(
                    self.rbr_dense(inputs) +
                    self.alpha * self.rbr_1x1(inputs) + id_out))
        else:
            return self.nonlinearity(
                self.se(
                    self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        """Derives the equivalent kernel and bias in a differentiable way.

        Returns:
            tuple: Equivalent kernel and bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(
                kernel1x1) + kernelid, bias3x3 + self.alpha * bias1x1 + biasid
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(
                kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad 1x1 tensor to 3x3.
        Args:
            kernel1x1 (Tensor): The input 1x1 kernel need to be padded.

        Returns:
            Tensor: 3x3 kernel after padded.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: nn.Module) -> Tuple[np.ndarray, Tensor]:
        """Derives the equivalent kernel and bias of a specific branch layer.

        Args:
            branch (nn.Module): The layer that needs to be equivalently
                transformed, which can be nn.Sequential or nn.Batchnorm2d

        Returns:
            tuple: Equivalent kernel and bias
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvModule):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, (nn.SyncBatchNorm, nn.BatchNorm2d))
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """Switch to deploy mode."""
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

class BottleRep(nn.Module):
    """Bottle Rep Block.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        block_cfg (dict): Config dict for the block used to build each
            layer. Defaults to dict(type='RepVGGBlock').
        adaptive_weight (bool): Add adaptive_weight when forward calculate.
            Defaults False.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 block_cfg: ConfigType = dict(type='RepVGGBlock'),
                 adaptive_weight: bool = False):
        super().__init__()
        conv1_cfg = block_cfg.copy()
        conv2_cfg = block_cfg.copy()

        conv1_cfg.update(
            dict(in_channels=in_channels, out_channels=out_channels))
        conv2_cfg.update(
            dict(in_channels=out_channels, out_channels=out_channels))

        self.conv1 = MODELS.build(conv1_cfg)
        self.conv2 = MODELS.build(conv2_cfg)

        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if adaptive_weight:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x: Tensor) -> Tensor:
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs

@MODELS.register_module()
class ConvWrapper(nn.Module):
    """Wrapper for normal Conv with SiLU activation.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): Conv bias. Default: True.
        norm_cfg (ConfigType): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (ConfigType): Config dict for activation layer.
            Defaults to dict(type='ReLU', inplace=True).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 norm_cfg: ConfigType = None,
                 act_cfg: ConfigType = dict(type='SiLU')):
        super().__init__()
        self.block = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=bias,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

class RepStageBlock(nn.Module):
    """RepStageBlock is a stage block with rep-style basic block.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        num_blocks (int, tuple[int]): Number of blocks.  Defaults to 1.
        bottle_block (nn.Module): Basic unit of RepStage.
            Defaults to RepVGGBlock.
        block_cfg (ConfigType): Config of RepStage.
            Defaults to 'RepVGGBlock'.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 1,
                 bottle_block: nn.Module = RepVGGBlock,
                 block_cfg: ConfigType = dict(type='RepVGGBlock')):
        super().__init__()
        block_cfg = block_cfg.copy()

        block_cfg.update(
            dict(in_channels=in_channels, out_channels=out_channels))

        self.conv1 = MODELS.build(block_cfg)

        block_cfg.update(
            dict(in_channels=out_channels, out_channels=out_channels))

        self.block = None
        if num_blocks > 1:
            self.block = nn.Sequential(*(MODELS.build(block_cfg)
                                         for _ in range(num_blocks - 1)))

        if bottle_block == BottleRep:
            self.conv1 = BottleRep(
                in_channels,
                out_channels,
                block_cfg=block_cfg,
                adaptive_weight=True)
            num_blocks = num_blocks // 2
            self.block = None
            if num_blocks > 1:
                self.block = nn.Sequential(*(BottleRep(
                    out_channels,
                    out_channels,
                    block_cfg=block_cfg,
                    adaptive_weight=True) for _ in range(num_blocks - 1)))

    def forward(self, x: Tensor) -> Tensor:
        """Forward process.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x

class BepC3StageBlock(nn.Module):
    """Beer-mug RepC3 Block.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        num_blocks (int): Number of blocks. Defaults to 1
        hidden_ratio (float): Hidden channel expansion.
            Default: 0.5
        concat_all_layer (bool): Concat all layer when forward calculate.
            Default: True
        block_cfg (dict): Config dict for the block used to build each
            layer. Defaults to dict(type='RepVGGBlock').
        norm_cfg (ConfigType): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (ConfigType): Config dict for activation layer.
            Defaults to dict(type='ReLU', inplace=True).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_blocks: int = 1,
                 hidden_ratio: float = 0.5,
                 concat_all_layer: bool = True,
                 block_cfg: ConfigType = dict(type='RepVGGBlock'),
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True)):
        super().__init__()
        hidden_channels = int(out_channels * hidden_ratio)

        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            2 * hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.block = RepStageBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_blocks=num_blocks,
            block_cfg=block_cfg,
            bottle_block=BottleRep)
        self.concat_all_layer = concat_all_layer
        if not concat_all_layer:
            self.conv3 = ConvModule(
                hidden_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, x):
        if self.concat_all_layer is True:
            return self.conv3(
                torch.cat((self.block(self.conv1(x)), self.conv2(x)), dim=1))
        else:
            return self.conv3(self.block(self.conv1(x)))

class CSPSPPFBottleneck(BaseModule):
    """The SPPF block having a CSP-like version in YOLOv6 3.0.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (int, tuple[int]): Sequential or number of kernel
            sizes of pooling layers. Defaults to 5.
        use_conv_first (bool): Whether to use conv before pooling layer.
            In YOLOv5 and YOLOX, the para set to True.
            In PPYOLOE, the para set to False.
            Defaults to True.
        mid_channels_scale (float): Channel multiplier, multiply in_channels
            by this amount to get mid_channels. This parameter is valid only
            when use_conv_fist=True.Defaults to 0.5.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None.
            which means using conv2d. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: Union[int, Sequence[int]] = 5,
                 use_conv_first: bool = True,
                 mid_channels_scale: float = 0.5,
                 conv_cfg: ConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)

        if use_conv_first:
            mid_channels = int(in_channels * mid_channels_scale)
            self.conv1 = ConvModule(
                in_channels,
                mid_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.conv3 = ConvModule(
                mid_channels,
                mid_channels,
                3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.conv4 = ConvModule(
                mid_channels,
                mid_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            mid_channels = in_channels
            self.conv1 = None
            self.conv3 = None
            self.conv4 = None

        self.conv2 = ConvModule(
            in_channels,
            mid_channels,
            1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.kernel_sizes = kernel_sizes

        if isinstance(kernel_sizes, int):
            self.poolings = nn.MaxPool2d(
                kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)
            conv2_in_channels = mid_channels * 4
        else:
            self.poolings = nn.ModuleList([
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ])
            conv2_in_channels = mid_channels * (len(kernel_sizes) + 1)

        self.conv5 = ConvModule(
            conv2_in_channels,
            mid_channels,
            1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv6 = ConvModule(
            mid_channels,
            mid_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv7 = ConvModule(
            mid_channels * 2,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        x0 = self.conv4(self.conv3(self.conv1(x))) if self.conv1 else x
        y = self.conv2(x)

        if isinstance(self.kernel_sizes, int):
            x1 = self.poolings(x0)
            x2 = self.poolings(x1)
            x3 = torch.cat([x0, x1, x2, self.poolings(x2)], dim=1)
        else:
            x3 = torch.cat(
                [x0] + [pooling(x0) for pooling in self.poolings], dim=1)

        x3 = self.conv6(self.conv5(x3))
        x = self.conv7(torch.cat([y, x3], dim=1))
        return x

class CSPSPPFModule(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels, out_channels, kernel_sizes=5, 
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 e=0.5):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = ConvModule(in_channels, c_, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv2 = ConvModule(in_channels, c_, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv3 = ConvModule(c_, c_, 3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv4 = ConvModule(c_, c_, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.m = nn.MaxPool2d(kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)
        self.cv5 = ConvModule(c_*4, c_, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv6 = ConvModule(c_, c_, 3, stride=1, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv7 =ConvModule(2*c_, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore')
        y1 = self.m(x1)
        y2 = self.m(y1)
        y3 = self.cv6(self.cv5(torch.cat([x1, y1, y2, self.m(y2)], 1)))
        result = self.cv7(torch.cat((y0, y3), dim=1))
        return result


class SPPFBottleneck(BaseModule):
    """Spatial pyramid pooling - Fast (SPPF) layer for
    YOLOv5, YOLOX and PPYOLOE by Glenn Jocher

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (int, tuple[int]): Sequential or number of kernel
            sizes of pooling layers. Defaults to 5.
        use_conv_first (bool): Whether to use conv before pooling layer.
            In YOLOv5 and YOLOX, the para set to True.
            In PPYOLOE, the para set to False.
            Defaults to True.
        mid_channels_scale (float): Channel multiplier, multiply in_channels
            by this amount to get mid_channels. This parameter is valid only
            when use_conv_fist=True.Defaults to 0.5.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None.
            which means using conv2d. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: Union[int, Sequence[int]] = 5,
                 use_conv_first: bool = True,
                 mid_channels_scale: float = 0.5,
                 conv_cfg: ConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg)

        if use_conv_first:
            mid_channels = int(in_channels * mid_channels_scale)
            self.conv1 = ConvModule(
                in_channels,
                mid_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            mid_channels = in_channels
            self.conv1 = None
        self.kernel_sizes = kernel_sizes
        if isinstance(kernel_sizes, int):
            self.poolings = nn.MaxPool2d(
                kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)
            conv2_in_channels = mid_channels * 4
        else:
            self.poolings = nn.ModuleList([
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ])
            conv2_in_channels = mid_channels * (len(kernel_sizes) + 1)

        self.conv2 = ConvModule(
            conv2_in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        if self.conv1:
            x = self.conv1(x)
        if isinstance(self.kernel_sizes, int):
            y1 = self.poolings(x)
            y2 = self.poolings(y1)
            x = torch.cat([x, y1, y2, self.poolings(y2)], dim=1)
        else:
            x = torch.cat(
                [x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x
    
class BiFusion(nn.Module):
    """BiFusion Block in YOLOv6.

    BiFusion fuses current-, high- and low-level features.
    Compared with concatenation in PAN, it fuses an extra low-level feature.

    Args:
        in_channels0 (int): The channels of current-level feature.
        in_channels1 (int): The input channels of lower-level feature.
        out_channels (int): The out channels of the BiFusion module.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels0: int,
                 in_channels1: int,
                 out_channels: int,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='ReLU', inplace=True)):
        super().__init__()
        self.conv1 = ConvModule(
            in_channels0,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = ConvModule(
            in_channels1,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv3 = ConvModule(
            out_channels * 3,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.upsample = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=2, stride=2, bias=True)
        self.downsample = ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: List[torch.Tensor]) -> Tensor:
        """Forward process
        Args:
            x (List[torch.Tensor]): The tensor list of length 3.
                x[0]: The high-level feature.
                x[1]: The current-level feature.
                x[2]: The low-level feature.
        """
        x0 = self.upsample(x[0])
        x1 = self.conv1(x[1])
        x2 = self.downsample(self.conv2(x[2]))
        return self.conv3(torch.cat((x0, x1, x2), dim=1))

'''----yolov8----'''
    
class MMDarknetBottleneck(BaseModule):
    """The basic bottleneck block used in Darknet.

    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (float): The kernel size of the convolution.
            Defaults to 0.5.
        add_identity (bool): Whether to add identity to the out.
            Defaults to True.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='Swish').
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='Swish'),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class DarknetBottleneck(MMDarknetBottleneck):
    """The basic bottleneck block used in Darknet.

    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
    The first convLayer has filter size of k1Xk1 and the second one has the
    filter size of k2Xk2.

    Note:
    This DarknetBottleneck is little different from MMDet's, we can
    change the kernel size and padding for each conv.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (float): The kernel size for hidden channel.
            Defaults to 0.5.
        kernel_size (Sequence[int]): The kernel size of the convolution.
            Defaults to (1, 3).
        padding (Sequence[int]): The padding size of the convolution.
            Defaults to (0, 1).
        add_identity (bool): Whether to add identity to the out.
            Defaults to True
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='Swish').
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 kernel_size: Sequence[int] = (1, 3),
                 padding: Sequence[int] = (0, 1),
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(in_channels, out_channels, init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        assert isinstance(kernel_size, Sequence) and len(kernel_size) == 2

        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            kernel_size[0],
            padding=padding[0],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            kernel_size[1],
            stride=1,
            padding=padding[1],
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity = \
            add_identity and in_channels == out_channels

@MODELS.register_module()
class CSPLayerWithTwoConv(BaseModule):
    """Cross Stage Partial Layer with 2 convolutions.

    Args:
        in_channels (int): The input channels of the CSP layer.
        out_channels (int): The output channels of the CSP layer.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        num_blocks (int): Number of blocks. Defaults to 1
        add_identity (bool): Whether to add identity in blocks.
            Defaults to True.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (:obj:`ConfigDict` or dict or list[dict] or
            list[:obj:`ConfigDict`], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.mid_channels = int(out_channels * expand_ratio)
        self.main_conv = ConvModule(
            in_channels,
            2 * self.mid_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.final_conv = ConvModule(
            (2 + num_blocks) * self.mid_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.blocks = nn.ModuleList(
            DarknetBottleneck(
                self.mid_channels,
                self.mid_channels,
                expansion=1,
                kernel_size=(3, 3),
                padding=(1, 1),
                add_identity=add_identity,
                use_depthwise=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks))

    def forward(self, x: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        return self.final_conv(torch.cat(x_main, 1))

'''----yoloworld----'''    

@MODELS.register_module()
class MaxSigmoidCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """Sigmoid-attention based CSP layer with two convolution layers."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            guide_channels: int,
            embed_channels: int,
            num_heads: int = 1,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            with_scale: bool = False,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            use_einsum: bool = True,
            depth:int =3) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        self.final_conv = ConvModule((3 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        self.attn_block = nn.ModuleList()
        for _ in range(depth):
            self.attn_block.append(MaxSigmoidAttnBlock(self.mid_channels,
                                              self.mid_channels,
                                              guide_channels=guide_channels,
                                              embed_channels=embed_channels,
                                              num_heads=num_heads,
                                              with_scale=with_scale,
                                              conv_cfg=conv_cfg,
                                              norm_cfg=norm_cfg,
                                              use_einsum=use_einsum))
        
       
    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_attn = x_main[-1]
        for attn in self.attn_block:
            x_attn = attn(x_attn, guide)
        x_main.append(x_attn)
        return self.final_conv(torch.cat(x_main, 1))

@MODELS.register_module()
class MaxSigmoidAttnBlock(BaseModule):
    """Max Sigmoid attention block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 guide_channels: int,
                 embed_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 use_depthwise: bool = False,
                 with_scale: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 init_cfg: OptMultiConfig = None,
                 use_einsum: bool = False) -> None:
        super().__init__(init_cfg=init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads
        self.use_einsum = False

        self.embed_conv = ConvModule(
            in_channels,
            embed_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None) if embed_channels != in_channels else None
        self.guide_fc = Linear(guide_channels, embed_channels)
        self.bias = nn.Parameter(torch.zeros(num_heads))
        if with_scale:
            self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        else:
            self.scale = 1.0

        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        B, _, H, W = x.shape

        guide = self.guide_fc(guide)
        guide = guide.reshape(B, -1, self.num_heads, self.head_channels)
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = embed.reshape(B, self.num_heads, self.head_channels, H, W)

        if self.use_einsum:
            attn_weight = torch.einsum('bmchw,bnmc->bmhwn', embed, guide)
        else:
            batch, m, channel, height, width = embed.shape
            _, n, _, _ = guide.shape
            embed = embed.permute(0, 1, 3, 4, 2)
            embed = embed.reshape(batch, m, -1, channel)
            guide = guide.permute(0, 2, 3, 1)
            attn_weight = torch.matmul(embed, guide)
            attn_weight = attn_weight.reshape(batch, m, height, width, n)

        attn_weight = attn_weight.max(dim=-1)[0]
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid() * self.scale

        x = self.project_conv(x)
        x = x.reshape(B, self.num_heads, -1, H, W)
        x = x * attn_weight.unsqueeze(2)
        x = x.reshape(B, -1, H, W)
        return x
    
@MODELS.register_module()
class ImagePoolingAttentionModule(nn.Module):

    def __init__(self,
                 image_channels: List[int],
                 text_channels: int,
                 embed_channels: int,
                 with_scale: bool = False,
                 num_feats: int = 3,
                 num_heads: int = 8,
                 pool_size: int = 3,
                 use_einsum: bool = True):
        super().__init__()

        self.text_channels = text_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.num_feats = num_feats
        self.head_channels = embed_channels // num_heads
        self.pool_size = pool_size
        self.use_einsum = use_einsum
        if with_scale:
            self.scale = nn.Parameter(torch.tensor([0.]), requires_grad=True)
        else:
            self.scale = 1.0
        self.projections = nn.ModuleList([
            ConvModule(in_channels, embed_channels, 1, act_cfg=None)
            for in_channels in image_channels
        ])
        self.query = nn.Sequential(nn.LayerNorm(text_channels),
                                   Linear(text_channels, embed_channels))
        self.key = nn.Sequential(nn.LayerNorm(embed_channels),
                                 Linear(embed_channels, embed_channels))
        self.value = nn.Sequential(nn.LayerNorm(embed_channels),
                                   Linear(embed_channels, embed_channels))
        self.proj = Linear(embed_channels, text_channels)

        self.image_pools = nn.ModuleList([
            nn.AdaptiveMaxPool2d((pool_size, pool_size))
            for _ in range(num_feats)
        ])

    def forward(self, text_features, image_features):
        B = image_features[0].shape[0]
        assert len(image_features) == self.num_feats
        num_patches = self.pool_size**2
        mlvl_image_features = [
            pool(proj(x)).view(B, -1, num_patches)
            for (x, proj, pool
                 ) in zip(image_features, self.projections, self.image_pools)
        ]
        mlvl_image_features = torch.cat(mlvl_image_features,
                                        dim=-1).transpose(1, 2)
        q = self.query(text_features)
        k = self.key(mlvl_image_features)
        v = self.value(mlvl_image_features)

        q = q.reshape(B, -1, self.num_heads, self.head_channels)
        k = k.reshape(B, -1, self.num_heads, self.head_channels)
        v = v.reshape(B, -1, self.num_heads, self.head_channels)
        if self.use_einsum:
            attn_weight = torch.einsum('bnmc,bkmc->bmnk', q, k)
        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 3, 1)
            attn_weight = torch.matmul(q, k)

        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = F.softmax(attn_weight, dim=-1)
        if self.use_einsum:
            x = torch.einsum('bmnk,bkmc->bnmc', attn_weight, v)
        else:
            v = v.permute(0, 2, 1, 3)
            x = torch.matmul(attn_weight, v)
            x = x.permute(0, 2, 1, 3)
        x = self.proj(x.reshape(B, -1, self.embed_channels))
        text_features = text_features.permute(1,0,2)
        result = x * self.scale + text_features
        result = result.permute(1,0,2)
        return result
    
    
@MODELS.register_module()
class RepConvMaxSigmoidAttnBlock(BaseModule):
    """Max Sigmoid attention block."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int,
                 guide_channels: int,
                 kernel_size: int = 3,
                 padding: int = 1,
                 num_heads: int = 1,
                 use_depthwise: bool = False,
                 with_scale: bool = False,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN',
                                             momentum=0.03,
                                             eps=0.001),
                 init_cfg: OptMultiConfig = None,
                 use_einsum: bool = True) -> None:
        super().__init__(init_cfg=init_cfg)
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        assert (out_channels % num_heads == 0 and
                embed_channels % num_heads == 0), \
            'out_channels and embed_channels should be divisible by num_heads.'
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads
        self.use_einsum = use_einsum

        self.embed_conv = ConvModule(
            in_channels,
            embed_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None) if embed_channels != in_channels else None
        self.bias = nn.Parameter(torch.zeros(num_heads))
        self.num_heads = num_heads
        self.split_channels = embed_channels // num_heads
        self.guide_convs = nn.ModuleList(
            nn.Conv2d(self.split_channels, guide_channels, 1, bias=False)
            for _ in range(num_heads))
        self.project_conv = conv(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=1,
                                 padding=padding,
                                 conv_cfg=conv_cfg,
                                 norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, x: Tensor, txt_feats: Tensor = None) -> Tensor:
        """Forward process."""
        B, C, H, W = x.shape

        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = list(embed.split(self.split_channels, 1))
        # Bx(MxN)xHxW (H*c=C, H: heads)
        attn_list = [conv(x) for conv, x in zip(self.guide_convs, embed)]
        attn_weight = torch.cat(attn_list, dim=1)
        # BxMxNxHxW
        attn_weight = attn_weight.view(B, self.num_heads, -1, H, W)
        # attn_weight = torch.stack(
        #     [conv(x) for conv, x in zip(self.guide_convs, embed)])
        # BxMxNxHxW -> BxMxHxW
        attn_weight = attn_weight.max(dim=2)[0] / (self.head_channels**0.5)
        attn_weight = (attn_weight + self.bias.view(1, -1, 1, 1)).sigmoid()
        # .transpose(0, 1)
        # BxMx1xHxW
        attn_weight = attn_weight[:, :, None]
        x = self.project_conv(x)
        # BxHxCxHxW
        x = x.view(B, self.num_heads, -1, H, W)
        x = x * attn_weight
        x = x.view(B, -1, H, W)
        return x
        

@MODELS.register_module()
class RepConvMaxSigmoidCSPLayerWithTwoConv(CSPLayerWithTwoConv):
    """Sigmoid-attention based CSP layer with two convolution layers."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            guide_channels: int,
            embed_channels: int,
            num_heads: int = 1,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            with_scale: bool = False,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            use_einsum: bool = True) -> None:
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         expand_ratio=expand_ratio,
                         num_blocks=num_blocks,
                         add_identity=add_identity,
                         conv_cfg=conv_cfg,
                         norm_cfg=norm_cfg,
                         act_cfg=act_cfg,
                         init_cfg=init_cfg)

        self.final_conv = ConvModule((3 + num_blocks) * self.mid_channels,
                                     out_channels,
                                     1,
                                     conv_cfg=conv_cfg,
                                     norm_cfg=norm_cfg,
                                     act_cfg=act_cfg)

        self.attn_block = RepConvMaxSigmoidAttnBlock(
            self.mid_channels,
            self.mid_channels,
            embed_channels=embed_channels,
            guide_channels=guide_channels,
            num_heads=num_heads,
            with_scale=with_scale,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            use_einsum=use_einsum)

    def forward(self, x: Tensor, guide: Tensor) -> Tensor:
        """Forward process."""
        x_main = self.main_conv(x)
        x_main = list(x_main.split((self.mid_channels, self.mid_channels), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.blocks)
        x_main.append(self.attn_block(x_main[-1], guide))
        return self.final_conv(torch.cat(x_main, 1))

"""yolov11"""

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__()
        
        mid_channels = int(out_channels * expand_ratio)  # hidden channels
        self.main_conv1 = ConvModule(
                        in_channels, 
                        mid_channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)
        self.main_conv2 = ConvModule(
                        in_channels, 
                        mid_channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)
        self.final_conv = ConvModule(
                        2*mid_channels, 
                        out_channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)
                        
        self.blocks = nn.Sequential(
            *(DarknetBottleneck(
                mid_channels,
                mid_channels,
                expansion=1,
                kernel_size=(3, 3),
                padding=(1, 1),
                add_identity=add_identity,
                use_depthwise=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.final_conv(torch.cat((self.blocks(self.main_conv1(x)), self.main_conv2(x)), 1))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None,
            kernel_size = 3) -> None:
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(in_channels, out_channels, expand_ratio,
                         num_blocks, add_identity)
        mid_channels = int(out_channels * expand_ratio)  # hidden channels
        self.blocks = nn.Sequential(
            *(DarknetBottleneck(
                mid_channels,
                mid_channels,
                expansion=1,
                kernel_size=(kernel_size, kernel_size),
                padding=(1, 1),
                add_identity=add_identity,
                use_depthwise=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(num_blocks)))


@MODELS.register_module()
class C3K2(CSPLayerWithTwoConv):
    """Cross Stage Partial Layer with 3 convolutions with 2 bottlenecks.
        Only YOLOv11-m,l,x are available!!! (which are all C2f blocks are replaced by C3k2)
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: float = 0.5,
            num_blocks: int = 1,
            add_identity: bool = True,  # shortcut
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(in_channels, out_channels, expand_ratio,
                         num_blocks, add_identity,
                         init_cfg=init_cfg)

        self.blocks = nn.ModuleList(
            C3k(
                self.mid_channels,
                self.mid_channels,
                num_blocks=2,
                add_identity=add_identity,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
                ) for _ in range(num_blocks))
        

class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5,
                 norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg = None):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = ConvModule(dim, h, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.proj = ConvModule(dim, dim, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.pe = ConvModule(dim, dim, 3, 1, 1, groups=dim, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True,
                norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg = None,
                ) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.ffn = nn.Sequential(ConvModule(c, c * 2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                                 ConvModule(c * 2, c, 1, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x

@MODELS.register_module()
class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5, 
                norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg = None):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = ConvModule(c1, 2 * self.c, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv2 = ConvModule(2 * self.c, c1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, 
                                          num_heads=self.c // 64, 
                                          norm_cfg=norm_cfg, 
                                          act_cfg=None) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
    

class AFF(nn.Module):

    def __init__(self, channels, r=4):
        super(AFF, self).__init__()
        mid_channels = int(channels // r)
        
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, y):
        xy = x + y
        xyl = self.local_att(xy)
        xyg = self.global_att(xy)
        xylg = xyl + xyg
        m = self.sigmoid(xylg)

        return 2*x*m + 2*y*(1-m)
    
class AFF_CSP(nn.Module):

    def __init__(self, channels, r=4):
        super(AFF_CSP, self).__init__()
        split_channels = int(channels//2) # 256
        self.split_channels = split_channels
        mid_channels = int(channels // r) # 64
        
        self.local_att = nn.Sequential(
            nn.Conv2d(split_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, split_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(split_channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(split_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, split_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(split_channels),
        )

        self.sigmoid = nn.Sigmoid()
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, split_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(split_channels),
            nn.SiLU(inplace=True))
        
        # ConvModule(
        #                 2*mid_channels, 
        #                 out_channels,
        #                 1,
        #                 conv_cfg=conv_cfg,
        #                 norm_cfg=norm_cfg,
        #                 act_cfg=act_cfg)
    
    def forward(self, x, y):
        
        x_main = list(x.split((self.split_channels, self.split_channels), 1))
        
        xy = x_main[0] + y
        xyl = self.local_att(xy)
        xyg = self.global_att(xy)
        xylg = xyl + xyg
        m = self.sigmoid(xylg)
        out = 2*x_main[0]*m + 2*y*(1-m)
        
        return self.final_conv(torch.cat((out,x_main[1]),1))
    
    
class iAFF(nn.Module):

    def __init__(self, channels=64, r=4):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        xy = x + y
        xyl = self.local_att(xy)
        xyg = self.global_att(xy)
        xylg = xyl + xyg
        m = self.sigmoid(xylg)
        xy1 = x * m + y * (1 - m)

        xyl2 = self.local_att2(xy1)
        xyg2 = self.global_att(xy1)
        xylg2 = xyl2 + xyg2
        m2 = self.sigmoid(xylg2)
        return x * m2 + y * (1 - m2)
    
class iAFF_CSP(nn.Module):

    def __init__(self, channels=64, r=4):
        super(iAFF_CSP, self).__init__()
        # inter_channels = int(channels // r)
        split_channels = int(channels//2) # 256
        self.split_channels = split_channels
        mid_channels = int(channels // r) # 64
        # self.local_att = nn.Sequential(
        #     nn.Conv2d(split_channels, mid_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.SiLU(inplace=True),
        #     nn.Conv2d(mid_channels, split_channels, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(split_channels),
        # )
        
        self.local_att = nn.Sequential(
            nn.Conv2d(split_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, split_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(split_channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(split_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, split_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(split_channels),
        )

        self.local_att2 = nn.Sequential(
            nn.Conv2d(split_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, split_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(split_channels),
        )

        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(split_channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, split_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(split_channels),
        )

        self.sigmoid = nn.Sigmoid()
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, split_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(split_channels),
            nn.SiLU(inplace=True))

    def forward(self, x, y):
        
        x_main = list(x.split((self.split_channels, self.split_channels), 1))
        xy = x_main[0] + y
        
        xyl = self.local_att(xy)
        xyg = self.global_att(xy)
        xylg = xyl + xyg
        m = self.sigmoid(xylg)
        xy1 = x_main[0]  * m + y * (1 - m)

        xyl2 = self.local_att2(xy1)
        xyg2 = self.global_att(xy1)
        xylg2 = xyl2 + xyg2
        m2 = self.sigmoid(xylg2)
    
        out = 2*x_main[0]*m2 + 2*y*(1-m2)
        
        return self.final_conv(torch.cat((out,x_main[1]),1))


'''CBAM'''

@MODELS.register_module()
class ChannelAttention(BaseModule):
    def __init__(self, channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(channels, channels // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(channels // ratio, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(BaseModule):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(BaseModule):
    def __init__(self, dim, r=16):
        super(CBAM, self).__init__()
        self.dim = dim
        self.r = r
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(channels=self.dim, ratio=self.r)

    def forward(self, x):
        output = self.ca(x)
        output = self.sa(output)
        return output + x

# @MODELS.register_module()  
class C2fCBAM(BaseModule):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.c2f = CSPLayerWithTwoConv(in_channels, out_channels, **kwargs)
        self.cbam = CBAM(out_channels)
    
    def forward(self, x):
        return self.cbam(self.c2f(x))

class ESE(BaseModule):
    def __init__(self,
                 in_channels: int,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.1, eps=1e-5),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True)):
        super().__init__()
        self.fc = nn.Conv2d(in_channels, in_channels, 1)
        self.sig = nn.Sigmoid()
        self.conv = ConvModule(
            in_channels,
            in_channels,
            1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self._init_weights()

    def _init_weights(self):
        """Init weights."""
        nn.init.normal_(self.fc.weight, mean=0, std=0.001)

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
         Args:
             feat (Tensor): The input tensor.
             avg_feat (Tensor): Average pooling feature tensor.
         """
        avg_feat = F.adaptive_avg_pool2d(x, (1,1))
        weight = self.sig(self.fc(avg_feat))
        return self.conv(x * weight)

    
class LSKModule(BaseModule):
    def __init__(self, in_channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels)
        self.conv_spatial = nn.Conv2d(in_channels, in_channels, 7, stride=1, padding=9, groups=in_channels, dilation=3)
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(in_channels//2, in_channels, 1)

    def forward(self, x):   
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn
    
class LSKAttention(BaseModule):
    def __init__(self, in_channels):
        super().__init__()

        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LSKModule(in_channels)
        self.proj_2 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class DWConv(BaseModule):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x
    
class Mlp(BaseModule):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

@MODELS.register_module()  
class LSKA(BaseModule):
    def __init__(self, in_channels, mlp_ratio=4., drop=0.1,drop_path=0.1, 
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, in_channels)[1]
        self.norm2 = build_norm_layer(norm_cfg, in_channels)[1]
        self.attn = LSKAttention(in_channels)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = Mlp(in_features=in_channels, hidden_features=mlp_hidden_dim,  drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x
    
@MODELS.register_module()  
class ESELSKA(BaseModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.lska = LSKA(**kwargs)
        self.ese = ESE(**kwargs)
        
    def forward(self, x):
        return self.lska(self.ese(x))
    
'''Attentional Fusion modules'''
        
class ASFFDown(BaseModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            conv_cfg: OptConfigType = None,
            norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: ConfigType = dict(type='SiLU', inplace=True),
            kernel_size = 7,
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__()
        self.conv1 = ConvModule(
                out_channels,
                out_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.conv2 = ConvModule(
                in_channels,
                out_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.proj1 = ConvModule(
                out_channels * 2,
                out_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        # self.attn = SpatialAttention()
        
        self.saconv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = self.conv1(x)
        y = self.conv2(y)
        out = self.proj1(torch.cat([x,y], dim=1))
        
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.saconv(out)
        att_map = self.sigmoid(out)
        return att_map*x + (1-att_map)*y
        
        # return self.sigmoid(x)


# class ASFFUp(BaseModule):
#     def __init__(self, **kwargs):
#         super().__init__()
#     def forward(self, x):
        # pass
    
class DASFF(BaseModule):
    def __init__(self, 
                in_channels: int,
                out_channels: int,
                conv_cfg: OptConfigType = None,
                norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg: ConfigType = dict(type='SiLU', inplace=True)
                ):
        super().__init__()
        n_heads = n_groups = 4
        n_head_channels = out_channels // n_heads
        
        self.deformable_attn = DAttention(n_heads, n_head_channels, n_groups)
        self.conv1 = ConvModule(
                out_channels,
                out_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.conv2 = ConvModule(
                in_channels,
                out_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.proj1 = ConvModule(
                out_channels * 2,
                out_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
    
    def forward(self, x, y):
        x = self.conv1(x)
        y = self.conv2(y)
        out = self.proj1(torch.cat([x,y], dim=1))
        return out + self.deformable_attn(out)[0]

class DCASFF(BaseModule):
    def __init__(self, 
                in_channels: int,
                out_channels: int,
                conv_cfg: OptConfigType = None,
                norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
                act_cfg: ConfigType = dict(type='SiLU', inplace=True)
                ):
        super().__init__()
        n_heads = n_groups = 4
        n_head_channels = out_channels // n_heads
        
        self.deformable_cattn = DCAttention(n_heads, n_head_channels, n_groups)
        self.conv1 = ConvModule(
                out_channels,
                out_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.conv2 = ConvModule(
                in_channels,
                out_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        # self.proj1 = ConvModule(
        #         out_channels * 2,
        #         out_channels,
        #         1,
        #         stride=1,
        #         conv_cfg=conv_cfg,
        #         norm_cfg=norm_cfg,
        #         act_cfg=act_cfg)
        self.alpha = nn.Parameter(torch.zeros(1, out_channels, 1, 1))  # P3 gate
        self.beta = nn.Parameter(torch.zeros(1, out_channels, 1, 1))   # P4 gate
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, y):
        x = self.conv1(x)
        y = self.conv2(y)
        out = self.deformable_cattn(x,y)[0] # [1] : pos -> visualization
        # out = self.proj1(torch.cat([x,y], dim=1))
        return self.sigmoid(self.alpha) * out + self.sigmoid(self.beta) * x
    
class DCAttention(nn.Module):
    def __init__(
        self,  n_heads, n_head_channels, n_groups,
        attn_drop = 0.0, proj_drop =0.0, stride = 4, 
        use_pe = True, dwc_pe = False, offset_range_factor = 2,
        no_off=False, fixed_pe=False, ksize=7,  log_cpb= False
    ):

        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = (128, 128)
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and not self.no_off:
            # if self.dwc_pe:
            #     self.rpe_table = nn.Conv2d(
            #         self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            # elif self.fixed_pe:
            #     self.rpe_table = nn.Parameter(
            #         torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
            #     )
            #     trunc_normal_(self.rpe_table, std=0.01)
            # elif self.log_cpb:
            #     # Borrowed from Swin-V2
            #     self.rpe_table = nn.Sequential(
            #         nn.Linear(2, 32, bias=True),
            #         nn.ReLU(inplace=True),
            #         nn.Linear(32, self.n_group_heads, bias=False)
            #     )
            # else:
            self.rpe_table = nn.Parameter(
                torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
            )
            trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            # indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref
    
    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            # indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref

    def forward(self, x, y):
        assert x.size() == y.size()
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        q = self.proj_q(x)
        
        kv_off = einops.rearrange(y, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(kv_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            y_sampled = F.avg_pool2d(y, kernel_size=self.stride, stride=self.stride)
            assert y_sampled.size(2) == Hk and y_sampled.size(3) == Wk, f"Size is {y_sampled.size()}"
        else:
            y_sampled = F.grid_sample(
                input=y.reshape(B * self.n_groups, self.n_group_channels, H, W), 
                grid=pos[..., (1, 0)], # y, x -> x, y
                mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
                

        y_sampled = y_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(y_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(y_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe and (not self.no_off):

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels, H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(4.0) # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement) # B * g, H * W, n_sample, h_g
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
                attn_bias = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads, g=self.n_groups),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True) # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        out = self.proj_drop(self.proj_out(out))

        return out, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)   

class DAttention(nn.Module):
    def __init__(
        self,  n_heads, n_head_channels, n_groups,
        attn_drop = 0.0, proj_drop =0.0, stride = 4, 
        use_pe = True, dwc_pe = False, offset_range_factor = 2,
        no_off=False, fixed_pe=False, ksize=7,  log_cpb= False
    ):

        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = (128, 128)
        # self.kv_h, self.kv_w = kv_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

        if self.use_pe and not self.no_off:
            # if self.dwc_pe:
            #     self.rpe_table = nn.Conv2d(
            #         self.nc, self.nc, kernel_size=3, stride=1, padding=1, groups=self.nc)
            # elif self.fixed_pe:
            #     self.rpe_table = nn.Parameter(
            #         torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
            #     )
            #     trunc_normal_(self.rpe_table, std=0.01)
            # elif self.log_cpb:
            #     # Borrowed from Swin-V2
            #     self.rpe_table = nn.Sequential(
            #         nn.Linear(2, 32, bias=True),
            #         nn.ReLU(inplace=True),
            #         nn.Linear(32, self.n_group_heads, bias=False)
            #     )
            # else:
            self.rpe_table = nn.Parameter(
                torch.zeros(self.n_heads, self.q_h * 2 - 1, self.q_w * 2 - 1)
            )
            trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            # indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref
    
    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            # indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2

        return ref

    def forward(self, x):

        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Hk and x_sampled.size(3) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, H, W), 
                grid=pos[..., (1, 0)], # y, x -> x, y
                mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
                

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale)

        if self.use_pe and (not self.no_off):

            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels, H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, n_sample)
            elif self.log_cpb:
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(4.0) # d_y, d_x [-8, +8]
                displacement = torch.sign(displacement) * torch.log2(torch.abs(displacement) + 1.0) / np.log2(8.0)
                attn_bias = self.rpe_table(displacement) # B * g, H * W, n_sample, h_g
                attn = attn + einops.rearrange(attn_bias, 'b m n h -> (b h) m n', h=self.n_group_heads)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                q_grid = self._get_q_grid(H, W, B, dtype, device)
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
                attn_bias = F.grid_sample(
                    input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=self.n_group_heads, g=self.n_groups),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True) # B * g, h_g, HW, Ns

                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)

        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y, pos.reshape(B, self.n_groups, Hk, Wk, 2), reference.reshape(B, self.n_groups, Hk, Wk, 2)
    
class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')