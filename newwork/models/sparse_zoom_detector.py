"""Detector wrapper with spatial supervision for the sparse text router."""

from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from mmdet.structures import OptSampleList, SampleList
from torch import Tensor

from mmyolo.registry import MODELS
from yolo_world.models.detectors.yolo_world import SimpleYOLOWorldDetector


@MODELS.register_module()
class SparseZoomYOLOWorldDetector(SimpleYOLOWorldDetector):
    """SimpleYOLOWorldDetector with GT-supervised P2 routing."""

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
        ce = F.binary_cross_entropy_with_logits(logits, targets,
                                                reduction='none')
        probs = logits.sigmoid()
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        return (alpha_t * (1 - pt).pow(gamma) * ce).mean()

    def _router_losses(self, batch_inputs: Tensor,
                       batch_data_samples: Union[list, dict]) -> Dict[str, Tensor]:
        state = self.neck.get_router_state()
        logits = state['saliency_logits']
        batch, _, feat_h, feat_w = logits.shape
        input_h, input_w = batch_inputs.shape[-2:]
        targets = logits.new_zeros((batch, 1, feat_h, feat_w))

        if not isinstance(batch_data_samples, dict):
            raise TypeError('Sparse router training expects yolow_collate dict.')
        packed_boxes = batch_data_samples['bboxes_labels']
        coverage_terms = []
        window_scores = state['window_scores']
        selected_scores = window_scores.gather(1, state['selected_indices'])
        route_threshold = selected_scores.min(dim=1).values.detach()
        num_window_cols = (feat_w + self.neck.window_size - 1) // self.neck.window_size
        for row in packed_boxes:
            image_idx = int(row[0].item())
            x1, y1, x2, y2 = row[-4:].to(logits)
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
            fx1 = int(torch.floor((cx - half_w) * feat_w / input_w).clamp(
                0, feat_w - 1).item())
            fy1 = int(torch.floor((cy - half_h) * feat_h / input_h).clamp(
                0, feat_h - 1).item())
            fx2 = int(torch.ceil((cx + half_w) * feat_w / input_w).clamp(
                fx1 + 1, feat_w).item())
            fy2 = int(torch.ceil((cy + half_h) * feat_h / input_h).clamp(
                fy1 + 1, feat_h).item())
            targets[image_idx, 0, fy1:fy2, fx1:fx2] = 1.0
            region_prob = logits[image_idx, 0, fy1:fy2, fx1:fx2].sigmoid()
            local_coverage = -torch.log(region_prob.max().clamp_min(1e-6))
            center_fx = min(feat_w - 1, max(0, (fx1 + fx2 - 1) // 2))
            center_fy = min(feat_h - 1, max(0, (fy1 + fy2 - 1) // 2))
            window_x = center_fx // self.neck.window_size
            window_y = center_fy // self.neck.window_size
            window_index = window_y * num_window_cols + window_x
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
        loss_coverage = loss_coverage * self.router_coverage_weight

        soft_active_ratio = logits.sigmoid().mean()
        target_ratio = float(self.neck.sparse_ratio)
        loss_budget = F.smooth_l1_loss(
            soft_active_ratio,
            soft_active_ratio.new_tensor(target_ratio))
        loss_budget = loss_budget * self.sparse_budget_weight
        return dict(
            loss_router=loss_router,
            loss_router_coverage=loss_coverage,
            loss_sparse_budget=loss_budget)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            losses = self.bbox_head.loss(img_feats, batch_data_samples)
        else:
            # Call loss_by_feat directly.  The upstream YOLOWorldHead.loss in
            # this repository omits ``txt_masks`` when assembling its fast
            # yolow_collate loss arguments, while loss_by_feat requires it.
            outs = self.bbox_head(img_feats, txt_feats, None)
            loss_inputs = outs + (None, batch_data_samples['bboxes_labels'],
                                  batch_data_samples['img_metas'])
            losses = self.bbox_head.loss_by_feat(*loss_inputs)
        losses.update(self._router_losses(batch_inputs, batch_data_samples))
        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        self.bbox_head.num_classes = self.num_test_classes
        if self.reparameterized:
            results_list = self.bbox_head.predict(
                img_feats, batch_data_samples, rescale=rescale)
        else:
            results_list = self.bbox_head.predict(
                img_feats, txt_feats, None, batch_data_samples,
                rescale=rescale)
        return self.add_pred_to_datasample(batch_data_samples, results_list)

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[list]:
        img_feats, txt_feats = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        if self.reparameterized:
            return self.bbox_head.forward(img_feats)
        return self.bbox_head.forward(img_feats, txt_feats, None)
