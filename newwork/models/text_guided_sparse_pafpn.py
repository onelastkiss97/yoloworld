"""Text-guided sparse P2 refinement on top of the original YOLO-World PAN.

The original P3--P5 ``YOLOWorldPAFPN`` is deliberately kept as a child with
the same parameter names.  A P2 fusion path and a text-conditioned window
router are added around it, which makes loading the released three-scale
YOLO-World checkpoint straightforward.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.utils import ConfigType, OptMultiConfig
from torch import Tensor

from mmyolo.models.layers import CSPLayerWithTwoConv
from mmyolo.models.utils import make_divisible, make_round
from mmyolo.registry import MODELS
from yolo_world.models.necks.yolo_world_pafpn import YOLOWorldPAFPN


@MODELS.register_module()
class DynamicTextGuidedRouter(nn.Module):
    """Compute dense image/text similarity logits for spatial routing."""

    def __init__(self,
                 in_channels: int,
                 text_channels: int = 512,
                 router_channels: int = 64,
                 logit_scale_init: float = 10.0) -> None:
        super().__init__()
        self.image_proj = nn.Conv2d(in_channels, router_channels, 1, bias=False)
        self.text_proj = nn.Linear(text_channels, router_channels, bias=True)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(logit_scale_init), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, image: Tensor, text: Tensor) -> Tensor:
        if text is None:
            raise ValueError('DynamicTextGuidedRouter requires text features.')
        if text.ndim == 2:
            text = text.unsqueeze(0)
        if text.shape[0] == 1 and image.shape[0] > 1:
            text = text.expand(image.shape[0], -1, -1)
        if text.shape[0] != image.shape[0]:
            raise ValueError('Image and text batch sizes must match.')

        image_embed = F.normalize(self.image_proj(image), dim=1, eps=1e-6)
        text_embed = F.normalize(self.text_proj(text), dim=-1, eps=1e-6)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return torch.einsum('bchw,bnc->bnhw', image_embed,
                            text_embed) * scale + self.bias


@MODELS.register_module()
class RepTextGuidedRouter(nn.Module):
    """Reparameterized router with prompt projections stored as a 1x1 conv."""

    def __init__(self,
                 in_channels: int,
                 num_prompts: int = 1,
                 router_channels: int = 64,
                 logit_scale_init: float = 10.0) -> None:
        super().__init__()
        self.image_proj = nn.Conv2d(in_channels, router_channels, 1, bias=False)
        self.guide_conv = nn.Conv2d(
            router_channels, num_prompts, 1, bias=False)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(logit_scale_init), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, image: Tensor, text: Optional[Tensor] = None) -> Tensor:
        image_embed = F.normalize(self.image_proj(image), dim=1, eps=1e-6)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return self.guide_conv(image_embed) * scale + self.bias


class SparseLocalRefinement(nn.Module):
    """Shared lightweight residual block evaluated on selected windows only."""

    def __init__(self,
                 channels: int,
                 num_blocks: int = 2,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True)) -> None:
        super().__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(
                DepthwiseSeparableConvModule(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.blocks = nn.Sequential(*layers)
        self.delta = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.delta(self.blocks(x))


@MODELS.register_module()
class TextGuidedSparseYOLOWorldPAFPN(YOLOWorldPAFPN):
    """YOLO-World PAN with a text-routed sparse high-resolution P2 path.

    Args:
        p2_in_channels: Canonical (pre-width-multiplier) backbone P2 channels.
        p2_out_channels: Canonical output channels of the new stride-4 level.
        reparameterized: Replace the dynamic text projection with fixed prompt
            convolution weights.
        window_size: Non-overlapping window size on the P2 feature map.
        sparse_ratio: Fraction of P2 windows refined per image.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 guide_channels: int,
                 embed_channels: List[int],
                 num_heads: List[int],
                 p2_in_channels: int = 128,
                 p2_out_channels: int = 128,
                 router_channels: int = 64,
                 text_channels: int = 512,
                 num_prompts: int = 1,
                 reparameterized: bool = False,
                 window_size: int = 8,
                 sparse_ratio: float = 0.25,
                 min_windows: int = 1,
                 refinement_blocks: int = 2,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 block_cfg: ConfigType = dict(type='CSPLayerWithTwoConv'),
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 init_cfg: OptMultiConfig = None) -> None:
        if not 0 < sparse_ratio <= 1:
            raise ValueError('sparse_ratio must be in (0, 1].')
        if window_size < 1:
            raise ValueError('window_size must be positive.')
        self.p2_in_channels = p2_in_channels
        self.p2_out_channels = p2_out_channels
        self.window_size = window_size
        self.sparse_ratio = sparse_ratio
        self.min_windows = min_windows
        self.reparameterized = reparameterized

        # Builds the unchanged P3--P5 PAN under the original parameter names.
        super().__init__(
            in_channels=in_channels,
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

        p2_in = make_divisible(p2_in_channels, widen_factor)
        p2_out = make_divisible(p2_out_channels, widen_factor)
        if isinstance(out_channels, int):
            p3_channels = make_divisible(out_channels, widen_factor)
        else:
            p3_channels = make_divisible(out_channels[0], widen_factor)

        self.p2_lateral = ConvModule(
            p2_in, p2_out, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.p3_to_p2 = ConvModule(
            p3_channels, p2_out, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.p2_fusion = CSPLayerWithTwoConv(
            p2_out * 2,
            p2_out,
            num_blocks=make_round(num_csp_blocks, deepen_factor),
            add_identity=False,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if reparameterized:
            self.router = RepTextGuidedRouter(
                p2_out, num_prompts, router_channels)
        else:
            self.router = DynamicTextGuidedRouter(
                p2_out, text_channels, router_channels)
        self.local_refinement = SparseLocalRefinement(
            p2_out, refinement_blocks, norm_cfg, act_cfg)
        self._router_state: Dict[str, Tensor] = {}

    def _sparse_refine(self, feature: Tensor,
                       saliency_logits: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Refine only hard Top-K windows; use soft gates for router gradients."""
        batch, channels, height, width = feature.shape
        ws = self.window_size
        pad_h = (ws - height % ws) % ws
        pad_w = (ws - width % ws) % ws
        padded = F.pad(feature, (0, pad_w, 0, pad_h))
        padded_logits = F.pad(
            saliency_logits, (0, pad_w, 0, pad_h), value=-20.0)
        padded_h, padded_w = padded.shape[-2:]

        window_scores = F.avg_pool2d(
            padded_logits, kernel_size=ws, stride=ws).flatten(1)
        num_windows = window_scores.shape[1]
        topk = min(num_windows,
                   max(self.min_windows,
                       int(math.ceil(num_windows * self.sparse_ratio))))
        selected_scores, selected_indices = torch.topk(
            window_scores, k=topk, dim=1, sorted=False)

        patches = F.unfold(padded, kernel_size=ws, stride=ws)
        gather_idx = selected_indices[:, None, :].expand(
            -1, channels * ws * ws, -1)
        selected = patches.gather(2, gather_idx)
        selected = selected.transpose(1, 2).reshape(
            batch * topk, channels, ws, ws)
        delta = self.local_refinement(selected)
        # The hard route selects compute; this gate sends detection gradients
        # into the router for the selected regions.
        gate = selected_scores.sigmoid().reshape(batch * topk, 1, 1, 1)
        delta = (delta * gate).reshape(batch, topk, channels * ws * ws)
        delta = delta.transpose(1, 2)

        sparse_patches = patches.new_zeros(patches.shape)
        sparse_patches.scatter_(2, gather_idx, delta)
        delta_map = F.fold(
            sparse_patches,
            output_size=(padded_h, padded_w),
            kernel_size=ws,
            stride=ws)
        refined = padded + delta_map

        selected_mask = window_scores.new_zeros(window_scores.shape)
        selected_mask.scatter_(1, selected_indices, 1.0)
        state = dict(
            window_scores=window_scores,
            selected_indices=selected_indices,
            selected_mask=selected_mask,
            active_ratio=selected_mask.mean())
        return refined[..., :height, :width], state

    def get_router_state(self) -> Dict[str, Tensor]:
        """Return tensors from the most recent forward for auxiliary losses."""
        return self._router_state

    def forward(self,
                img_feats: List[Tensor],
                txt_feats: Optional[Tensor] = None) -> tuple:
        if len(img_feats) != 4:
            raise ValueError('Expected backbone features (P2, P3, P4, P5).')

        p2_backbone = img_feats[0]
        p3_to_p5 = img_feats[1:]
        base_outs = super().forward(p3_to_p5, txt_feats)

        p2_low = self.p2_lateral(p2_backbone)
        p3_high = self.p3_to_p2(base_outs[0])
        p3_high = F.interpolate(
            p3_high, size=p2_low.shape[-2:], mode='nearest')
        p2 = self.p2_fusion(torch.cat((p2_low, p3_high), dim=1))

        prompt_logits = self.router(p2, txt_feats)
        saliency_logits = prompt_logits.max(dim=1, keepdim=True).values
        p2, sparse_state = self._sparse_refine(p2, saliency_logits)
        self._router_state = dict(
            prompt_logits=prompt_logits,
            saliency_logits=saliency_logits,
            **sparse_state)
        return (p2, ) + base_outs

