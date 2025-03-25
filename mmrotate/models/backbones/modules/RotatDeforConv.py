import einops
import math
import torch
import torch.nn as nn
from mmcv.ops import DeformConv2d
from .weight_init import trunc_normal_
from mmengine.model import BaseModule
from mmrotate.models.blocks import *

class LayerNormProxy(nn.Module):
    # copy from https://github.com/LeapLabTHU/DAT/blob/main/models/dat_blocks.py
    def __init__(self, dim,
                 ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')
    
class RotationallyDeformableConvolution(nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
                
        # dist angle추출  
        self.dwc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                             groups=in_channels, bias=False)
        self.norm = LayerNormProxy(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.fc_dist = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.dist_act_func = nn.ReLU(inplace=True)
        
        self.fc_angle = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.ang_act_func = nn.Softsign()
        
        
        # Deformable Convolution 레이어 초기화
        self.deformconv = DeformConv2d(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride,
                                       padding,
                                       dilation=1)
        # init weights
        trunc_normal_(self.dwc.weight, std=.02)
        trunc_normal_(self.fc_dist.weight, std=.02)
        trunc_normal_(self.fc_angle.weight, std=.02)
        
    def get_dist_angle(self, x):
        x = self.dwc(x)
        x = self.norm(x)
        x = self.relu(x)
        
        r = self.fc_dist(x)
        r = self.dist_act_func(r)
                
        theta = self.fc_angle(x)
        theta = self.ang_act_func(theta) * (math.pi / 180)  # degrees to radians
        return r, theta

    def forward(self, input):
        B, C, H, W = input.shape
        
        # 거리와 각도 추출
        r, theta = self.get_dist_angle(input)
        
        # offsets 계산
        offset_x = r * torch.cos(theta).repeat(1, self.kernel_size * self.kernel_size, 1, 1)
        offset_y = r * torch.sin(theta).repeat(1, self.kernel_size * self.kernel_size, 1, 1)

        # offsets 결합
        offsets = torch.cat([offset_x, offset_y], dim=1)  # dim=1에서 합칩니다.
        
        # Deformable Convolution 적용
        output = self.deformconv(input, offsets)
        return output

if __name__ == '__main__':
    x = torch.rand((1,3,3,3))
    rdc = RotationallyDeformableConvolution(3, 16, 3, 1, 1, x.shape)
    out = rdc(x)
    print(out.shape)