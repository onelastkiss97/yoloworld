#!/usr/bin/env python3
"""Build the configured model and smoke-test P2--P5 feature extraction."""

from pathlib import Path

import torch
from mmengine.config import Config
from mmyolo.registry import MODELS
from mmyolo.utils import register_all_modules


def main():
    register_all_modules()
    import newwork  # noqa: F401

    root = Path(__file__).resolve().parents[2]
    cfg = Config.fromfile(
        root / 'newwork/configs/sparse_zoom_yoloworld_v2_m_drone_1280.py')
    model = MODELS.build(cfg.model)
    model.eval()
    with torch.no_grad():
        feats, text = model.extract_feat(torch.randn(1, 3, 256, 256), None)
    assert [tuple(x.shape[-2:]) for x in feats] == [
        (64, 64), (32, 32), (16, 16), (8, 8)
    ]
    assert text.shape == (1, 1, 512)
    state = model.neck.get_router_state()
    assert state['saliency_logits'].shape[-2:] == (64, 64)
    assert abs(float(state['active_ratio']) - 0.25) < 1e-6
    print('Model build and P2--P5 smoke test passed.')


if __name__ == '__main__':
    main()
