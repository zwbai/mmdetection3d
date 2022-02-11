# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner import auto_fp16
from torch import nn

from ..builder import FUSION_ENCODERS


@FUSION_ENCODERS.register_module()
class PillarGridFusion(nn.Module):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, fusion_channels):
        super().__init__()
        # self.in_channels = in_channels
        # self.fp16_enabled = False
        self.maxpool3D = nn.MaxPool3d([1, 1, fusion_channels], [1, 1, 1])

    # @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, x):
        """Foraward function to scatter features."""
        # TODO: rewrite the function in a batch manner
        # no need to deal with different batch cases
        x_fused = self.maxpool3D(x)
        x_squeeze = torch.squeeze(x_fused, 4)

        # print('x_fused', x_fused[:,63, 250, 200:250, :])
        # print('x_squeeze', x_squeeze[:,63, 250, 200:250])
        return x_squeeze
