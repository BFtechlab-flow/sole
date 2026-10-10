"""Backward-compatible metric imports.

Use :mod:`src.pq_official` directly in new code.
"""
from .pq_official import (
    aggregate_scores,
    bootstrap_macro_pq,
    bootstrap_micro_pq,
    evaluate_frames,
    iou_matrix,
    micro_pq,
    paired_bootstrap_micro_pq,
    pq_score,
)

# Old name kept for scripts written against V2/V3.
pairwise_iou = iou_matrix

__all__ = [
    "aggregate_scores",
    "bootstrap_macro_pq",
    "bootstrap_micro_pq",
    "evaluate_frames",
    "iou_matrix",
    "micro_pq",
    "pairwise_iou",
    "paired_bootstrap_micro_pq",
    "pq_score",
]
