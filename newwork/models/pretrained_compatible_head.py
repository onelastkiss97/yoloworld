"""Four-level YOLO-World heads that retain released V2-M tower widths."""

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmyolo.registry import MODELS
from yolo_world.models.dense_heads.yolo_world_head import (
    BNContrastiveHead, ContrastiveHead, YOLOWorldHeadModule)


@MODELS.register_module()
class PretrainedCompatibleYOLOWorldHeadModule(YOLOWorldHeadModule):
    """YOLO-World head with explicit hidden widths.

    Upstream YOLOv8 derives the shared classification width from level zero.
    Adding a lightweight P2 level would therefore change the P3--P5 tower
    width from 192 to 96 in YOLO-World V2-M and make released weights
    incompatible.  Explicit widths keep P2 lightweight while preserving all
    three pretrained detection towers exactly.
    """

    def __init__(self,
                 *args,
                 cls_hidden_channels: int = 192,
                 reg_hidden_channels: int = 64,
                 **kwargs) -> None:
        self.cls_hidden_channels = cls_hidden_channels
        self.reg_hidden_channels = reg_hidden_channels
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_contrasts = nn.ModuleList()

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(
                        self.in_channels[i],
                        self.reg_hidden_channels,
                        3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    ConvModule(
                        self.reg_hidden_channels,
                        self.reg_hidden_channels,
                        3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    nn.Conv2d(self.reg_hidden_channels, 4 * self.reg_max, 1)))
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(
                        self.in_channels[i],
                        self.cls_hidden_channels,
                        3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    ConvModule(
                        self.cls_hidden_channels,
                        self.cls_hidden_channels,
                        3,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg),
                    nn.Conv2d(self.cls_hidden_channels, self.embed_dims, 1)))
            if self.use_bn_head:
                contrast = BNContrastiveHead(
                    self.embed_dims,
                    self.norm_cfg,
                    use_einsum=self.use_einsum)
            else:
                contrast = ContrastiveHead(
                    self.embed_dims, use_einsum=self.use_einsum)
            self.cls_contrasts.append(contrast)

        self.register_buffer(
            'proj', torch.arange(self.reg_max, dtype=torch.float),
            persistent=False)
        if self.freeze_all:
            self._freeze_all()

