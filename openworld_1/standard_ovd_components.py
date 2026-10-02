"""Standard MMDetection loss path and early loss sanity check for YOLO-World."""

import math
from typing import Sequence, Tuple, Union

from mmengine.hooks import Hook
from mmdet.models.utils import unpack_gt_instances
from mmyolo.registry import HOOKS, MODELS
from torch import Tensor

from yolo_world.models.dense_heads.yolo_world_head import YOLOWorldHead
from yolo_world.models.detectors.yolo_world import YOLOWorldDetector


@MODELS.register_module()
class FrozenBackboneYOLOWorldDetector(YOLOWorldDetector):
    """YOLO-World detector that freezes both backbones during construction."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for parameter in self.backbone.image_model.parameters():
            parameter.requires_grad = False
        for parameter in self.backbone.text_model.parameters():
            parameter.requires_grad = False


@MODELS.register_module()
class StandardYOLOWorldHead(YOLOWorldHead):
    """YOLO-World head using standard ``list[DetDataSample]`` batches."""

    def loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             txt_masks: Tensor, batch_data_samples: Union[list, dict]) -> dict:
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = \
            unpack_gt_instances(batch_data_samples)
        outs = self(img_feats, txt_feats, txt_masks)
        return self.loss_by_feat(*outs, txt_masks, batch_gt_instances,
                                 batch_img_metas, batch_gt_instances_ignore)


@HOOKS.register_module()
class PositiveDetectionLossHook(Hook):
    """Fail early if a supposedly labelled run has no finite detection loss."""

    priority = 'VERY_HIGH'

    def __init__(self, check_iters: int = 25) -> None:
        self.check_iters = check_iters
        self.has_positive_bbox_loss = False
        self.has_positive_dfl_loss = False

    def after_train_iter(self, runner, batch_idx: int,
                         data_batch: Sequence = None, outputs: dict = None) -> None:
        if outputs is None:
            return
        values = {}
        for name in ('loss_cls', 'loss_bbox', 'loss_dfl'):
            value = outputs.get(name)
            if value is None:
                raise RuntimeError(f'Missing {name} from train-step outputs')
            value = float(value.detach().cpu()) if hasattr(value, 'detach') else float(value)
            if not math.isfinite(value):
                raise RuntimeError(f'Non-finite {name} at iteration {batch_idx}: {value}')
            values[name] = value
        self.has_positive_bbox_loss |= values['loss_bbox'] > 0.0
        self.has_positive_dfl_loss |= values['loss_dfl'] > 0.0
        if batch_idx + 1 == self.check_iters:
            if not self.has_positive_bbox_loss or not self.has_positive_dfl_loss:
                raise RuntimeError(
                    'No positive regression/DFL loss in the first '
                    f'{self.check_iters} iterations; aborting invalid run.')


@HOOKS.register_module()
class TrainableParameterAuditHook(Hook):
    """Abort unless trainable parameters are confined to approved prefixes."""

    priority = 'HIGHEST'

    def __init__(self, allowed_prefixes=('neck.', 'bbox_head.')) -> None:
        self.allowed_prefixes = tuple(allowed_prefixes)

    def before_train(self, runner) -> None:
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        trainable = [name for name, parameter in model.named_parameters()
                     if parameter.requires_grad]
        unexpected = [name for name in trainable
                      if not name.startswith(self.allowed_prefixes)]
        if not trainable or unexpected:
            raise RuntimeError(
                'Trainable-parameter audit failed. '
                f'trainable={len(trainable)}, unexpected={unexpected[:20]}')
        total = sum(parameter.numel() for parameter in model.parameters())
        active = sum(parameter.numel() for parameter in model.parameters()
                     if parameter.requires_grad)
        runner.logger.info(
            'Trainable-parameter audit passed: %d tensors, %d/%d parameters '
            '(%.2f%%), prefixes=%s', len(trainable), active, total,
            100.0 * active / total, self.allowed_prefixes)
