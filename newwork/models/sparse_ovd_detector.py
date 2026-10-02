"""Dynamic-text sparse high-resolution detector for standard OVD batches."""

from typing import Dict, Union

import torch
import torch.nn.functional as F
from mmdet.models.utils import unpack_gt_instances
from mmdet.structures import SampleList
from torch import Tensor

from mmyolo.registry import MODELS
from yolo_world.models.detectors.yolo_world import YOLOWorldDetector


@MODELS.register_module()
class TextGuidedSparseOVDDetector(YOLOWorldDetector):
    """YOLOWorldDetector with supervised text-guided P2 window routing."""

    def __init__(self,
                 *args,
                 router_loss_weight: float = 1.0,
                 router_coverage_weight: float = 0.5,
                 sparse_budget_weight: float = 0.1,
                 router_focal_alpha: float = 0.75,
                 router_focal_gamma: float = 2.0,
                 small_object_max_size: float = 96.0,
                 router_target_expand: float = 1.5,
                 **kwargs) -> None:
        self.router_loss_weight = router_loss_weight
        self.router_coverage_weight = router_coverage_weight
        self.sparse_budget_weight = sparse_budget_weight
        self.router_focal_alpha = router_focal_alpha
        self.router_focal_gamma = router_focal_gamma
        self.small_object_max_size = small_object_max_size
        self.router_target_expand = router_target_expand
        super().__init__(*args, **kwargs)

    @staticmethod
    def _focal_bce(logits: Tensor, targets: Tensor, alpha: float,
                   gamma: float) -> Tensor:
        ce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        probs = logits.sigmoid()
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        return (alpha_t * (1 - pt).pow(gamma) * ce).mean()

    def _router_losses(self, batch_inputs: Tensor,
                       batch_data_samples: SampleList) -> Dict[str, Tensor]:
        state = self.neck.get_router_state()
        logits = state['saliency_logits']
        batch, _, feat_h, feat_w = logits.shape
        input_h, input_w = batch_inputs.shape[-2:]
        targets = logits.new_zeros((batch, 1, feat_h, feat_w))

        window_scores = state['window_scores']
        selected_scores = window_scores.gather(1, state['selected_indices'])
        route_threshold = selected_scores.min(dim=1).values.detach()
        window_size = self.neck.window_size
        num_window_cols = (feat_w + window_size - 1) // window_size
        coverage_terms = []

        for image_idx, sample in enumerate(batch_data_samples):
            bboxes = sample.gt_instances.bboxes
            if hasattr(bboxes, 'tensor'):
                bboxes = bboxes.tensor
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox.to(logits)
                box_w = (x2 - x1).clamp(min=0)
                box_h = (y2 - y1).clamp(min=0)
                if torch.sqrt(box_w * box_h) > self.small_object_max_size:
                    continue

                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                half_w = torch.maximum(
                    box_w * self.router_target_expand / 2,
                    box_w.new_tensor(input_w / feat_w))
                half_h = torch.maximum(
                    box_h * self.router_target_expand / 2,
                    box_h.new_tensor(input_h / feat_h))
                fx1 = int(torch.floor(
                    (cx - half_w) * feat_w / input_w).clamp(
                        0, feat_w - 1).item())
                fy1 = int(torch.floor(
                    (cy - half_h) * feat_h / input_h).clamp(
                        0, feat_h - 1).item())
                fx2 = int(torch.ceil(
                    (cx + half_w) * feat_w / input_w).clamp(
                        fx1 + 1, feat_w).item())
                fy2 = int(torch.ceil(
                    (cy + half_h) * feat_h / input_h).clamp(
                        fy1 + 1, feat_h).item())

                targets[image_idx, 0, fy1:fy2, fx1:fx2] = 1.0
                region_prob = logits[
                    image_idx, 0, fy1:fy2, fx1:fx2].sigmoid()
                local_coverage = -torch.log(
                    region_prob.max().clamp_min(1e-6))
                center_fx = min(feat_w - 1,
                                max(0, (fx1 + fx2 - 1) // 2))
                center_fy = min(feat_h - 1,
                                max(0, (fy1 + fy2 - 1) // 2))
                window_index = ((center_fy // window_size) * num_window_cols
                                + center_fx // window_size)
                gt_window_score = window_scores[image_idx, window_index]
                ranking_coverage = F.softplus(
                    route_threshold[image_idx] - gt_window_score)
                coverage_terms.append(local_coverage + ranking_coverage)

        loss_router = self._focal_bce(
            logits, targets, self.router_focal_alpha,
            self.router_focal_gamma) * self.router_loss_weight
        if coverage_terms:
            loss_coverage = torch.stack(coverage_terms).mean()
        else:
            loss_coverage = logits.sum() * 0.0
        loss_coverage *= self.router_coverage_weight

        soft_ratio = logits.sigmoid().mean()
        target_ratio = soft_ratio.new_tensor(float(self.neck.sparse_ratio))
        loss_budget = F.smooth_l1_loss(soft_ratio, target_ratio)
        loss_budget *= self.sparse_budget_weight
        return dict(
            loss_router=loss_router,
            loss_router_coverage=loss_coverage,
            loss_sparse_budget=loss_budget)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        self.bbox_head.num_classes = self.num_train_classes
        img_feats, txt_feats, txt_masks = self.extract_feat(
            batch_inputs, batch_data_samples)
        gt_instances, ignored_instances, img_metas = unpack_gt_instances(
            batch_data_samples)
        outs = self.bbox_head(img_feats, txt_feats, txt_masks)
        losses = self.bbox_head.loss_by_feat(
            *outs, txt_masks, gt_instances, img_metas, ignored_instances)
        losses.update(self._router_losses(batch_inputs, batch_data_samples))
        return losses


@MODELS.register_module()
class FrozenBackboneTextGuidedSparseOVDDetector(TextGuidedSparseOVDDetector):
    """Sparse OVD detector with both backbones frozen before optimization.

    This mirrors the five-shot protocol in ``HANDOFF.md`` while retaining the
    dynamic text branch, P2 path, and router.  Freezing in the constructor is
    important because the optimizer is built only afterwards.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for parameter in self.backbone.image_model.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.text_model.parameters():
            parameter.requires_grad = False
