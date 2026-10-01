"""Backward-compatible metric imports.

Use :mod:`src.pq_official` directly in new code.
"""
from .pq_official import aggregate_scores, bootstrap_macro_pq, pairwise_iou, pq_score

__all__ = ["pairwise_iou", "pq_score", "aggregate_scores", "bootstrap_macro_pq"]
