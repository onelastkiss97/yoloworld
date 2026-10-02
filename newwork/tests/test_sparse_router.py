"""Small unit tests for routing and sparse window reconstruction."""

import torch

from newwork.models.text_guided_sparse_pafpn import (
    DynamicTextGuidedRouter, RepTextGuidedRouter)


def test_dynamic_router_shape_and_gradient():
    router = DynamicTextGuidedRouter(32, text_channels=16, router_channels=8)
    image = torch.randn(2, 32, 20, 24, requires_grad=True)
    text = torch.randn(2, 3, 16)
    logits = router(image, text)
    assert logits.shape == (2, 3, 20, 24)
    logits.mean().backward()
    assert image.grad is not None
    assert router.text_proj.weight.grad is not None


def test_rep_router_shape():
    router = RepTextGuidedRouter(32, num_prompts=3, router_channels=8)
    logits = router(torch.randn(2, 32, 20, 24))
    assert logits.shape == (2, 3, 20, 24)

