"""Model registrations for the sparse high-resolution extension."""

from .sparse_zoom_detector import SparseZoomYOLOWorldDetector
from .sparse_ovd_detector import (
    FrozenBackboneTextGuidedSparseOVDDetector,
    TextGuidedSparseOVDDetector,
)
from .pretrained_compatible_head import (
    PretrainedCompatibleYOLOWorldHeadModule)
from .text_guided_sparse_pafpn import (
    DynamicTextGuidedRouter, RepTextGuidedRouter,
    TextGuidedSparseYOLOWorldPAFPN)

__all__ = [
    'DynamicTextGuidedRouter', 'RepTextGuidedRouter',
    'TextGuidedSparseYOLOWorldPAFPN', 'SparseZoomYOLOWorldDetector',
    'TextGuidedSparseOVDDetector',
    'FrozenBackboneTextGuidedSparseOVDDetector',
    'PretrainedCompatibleYOLOWorldHeadModule'
]
