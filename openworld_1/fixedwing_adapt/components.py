"""Experiment-local adaptation and explicit per-class AP50 reporting."""
import tempfile
from pathlib import Path

import numpy as np
import torch
from mmdet.evaluation.metrics import CocoMetric
from mmyolo.registry import METRICS, MODELS
from pycocotools.cocoeval import COCOeval

from openworld_1.standard_ovd_components import FrozenBackboneYOLOWorldDetector


@METRICS.register_module()
class FixedWingCocoMetric(CocoMetric):
    def compute_metrics(self, results):
        metrics = super().compute_metrics(results)
        with tempfile.TemporaryDirectory() as tmp:
            files = self.results2json([r[1] for r in results], str(Path(tmp) / 'pred'))
            import json
            if not json.loads(Path(files['bbox']).read_text()):
                metrics['fixedwing_ap50'] = 0.0
                return metrics
            detections = self._coco_api.loadRes(files['bbox'])
            evaluator = COCOeval(self._coco_api, detections, 'bbox')
            evaluator.params.catIds = self.cat_ids
            evaluator.params.imgIds = self.img_ids
            evaluator.params.maxDets = list(self.proposal_nums)
            evaluator.params.iouThrs = self.iou_thrs
            evaluator.evaluate()
            evaluator.accumulate()
            precision = evaluator.eval['precision']
            iou50 = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, .5))[0])
            for index, cid in enumerate(evaluator.params.catIds):
                name = self._coco_api.loadCats([cid])[0]['name']
                values = precision[iou50, :, index, 0, -1]
                values = values[values > -1]
                value = float(values.mean()) if values.size else float('nan')
                key = 'fixedwing_ap50' if name == 'fixed-wing aircraft' else name.replace(' ', '_') + '_ap50'
                metrics[key] = value
        return metrics


@MODELS.register_module()
class FewShotYOLOWorldDetector(FrozenBackboneYOLOWorldDetector):
    """Freeze image/text encoders and BN running stats during few-shot fitting."""
    def __init__(self, *args, tune_neck=True, **kwargs):
        super().__init__(*args, **kwargs)
        if not tune_neck:
            for parameter in self.neck.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        # Batch size four and a repeated support set must not overwrite base BN statistics.
        for module in self.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self
