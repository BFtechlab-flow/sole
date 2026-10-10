"""Figures for the judging rubric and the technical report.

* :func:`plot_overlap_distribution` - distribution of IoU (or Dice) over every
  overlapping (GT, prediction) pair, the quantities the organizers plot;
* :func:`plot_relation_counts` - how many predictions overlap each GT filament
  (1:n, fragmentation) and how many GT filaments overlap each prediction
  (n:1, merging), from the degree histograms of ``src.pq_official``;
* :func:`overlay` - predicted instances (and optionally one annotator's
  filaments) drawn on the H-alpha image, for the qualitative part.

matplotlib is imported lazily so that the training code does not need it.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def _plt():
    import matplotlib

    if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
        try:
            matplotlib.use("Agg", force=False)
        except Exception:  # noqa: BLE001 - keep the active interactive backend
            pass
    import matplotlib.pyplot as plt

    return plt


def plot_overlap_distribution(values: Sequence[float], title: str, xlabel: str, ax=None, bins: int = 50):
    """Normalised histogram of the non-zero overlap scores, with the mean marked."""
    plt = _plt()
    ax = ax or plt.subplots(figsize=(5, 3.2))[1]
    v = np.asarray(list(values), dtype=float)
    v = v[v > 0]
    if v.size:
        weights = np.full(v.size, 1.0 / v.size)
        ax.hist(v, bins=np.linspace(0.0, 1.0, bins + 1), weights=weights, color="#2b5c8a", edgecolor="white",
                linewidth=0.4)
        ax.axvline(v.mean(), color="#c0392b", linestyle="--", linewidth=1.2, label=f"mean = {v.mean():.3f}")
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "no overlapping pairs", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("fraction of pairs")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot_relation_counts(gt_degree: Mapping, pred_degree: Mapping, ax=None, title: str = "GT / prediction relations"):
    """Bars right of zero: GT filaments by number of overlapping predictions
    (0 = missed, 1 = clean, >=2 = fragmented). Bars left of zero: predictions by
    number of overlapping GT filaments (0 = spurious, >=2 = merged)."""
    plt = _plt()
    ax = ax or plt.subplots(figsize=(6, 3.2))[1]
    gt = {int(k): int(v) for k, v in gt_degree.items()}
    pr = {int(k): int(v) for k, v in pred_degree.items()}

    def colour(d, side):
        if d == 0:
            return "#c0392b" if side == "gt" else "#e67e22"
        return "#27ae60" if d == 1 else "#8e44ad"

    for d, n in sorted(gt.items()):
        ax.bar(d + 0.5, n, width=0.8, color=colour(d, "gt"))
    for d, n in sorted(pr.items()):
        ax.bar(-(d + 0.5), n, width=0.8, color=colour(d, "pred"))
    ticks = [d + 0.5 for d in sorted(gt)] + [-(d + 0.5) for d in sorted(pr)]
    labels = [f"1:{d}" for d in sorted(gt)] + [f"{d}:1" for d in sorted(pr)]
    ax.set_xticks(ticks, labels, rotation=45, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    n_gt, n_pr = sum(gt.values()), sum(pr.values())
    ax.text(0.99, 0.97, f"GT: {n_gt} (missed {gt.get(0, 0)}, fragmented {sum(v for k, v in gt.items() if k >= 2)})",
            ha="right", va="top", transform=ax.transAxes, fontsize=7)
    ax.text(0.01, 0.97, f"pred: {n_pr} (spurious {pr.get(0, 0)}, merging {sum(v for k, v in pr.items() if k >= 2)})",
            ha="left", va="top", transform=ax.transAxes, fontsize=7)
    return ax


def _contours(mask: np.ndarray) -> np.ndarray:
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0


def overlay(gray: np.ndarray, labels: np.ndarray | None = None, gt_masks: Sequence[np.ndarray] | None = None,
            fill_alpha: float = 0.35, seed: int = 0, gt_color=(1.0, 0.8, 0.0)) -> np.ndarray:
    """RGB uint8 image: H-alpha in grey, predicted instances as coloured fills
    with contours, GT filaments (if given) as contours in ``gt_color`` (amber,
    readable on both the bright disk and the dark background)."""
    g = np.asarray(gray, dtype=np.float32)
    g = (g - g.min()) / max(float(g.max() - g.min()), 1e-6)
    rgb = np.repeat(g[..., None], 3, axis=2)
    if labels is not None and labels.max() > 0:
        rng = np.random.default_rng(seed)
        palette = rng.uniform(0.25, 1.0, size=(int(labels.max()) + 1, 3))
        palette[:, int(rng.integers(0, 3))] = 1.0
        fg = labels > 0
        rgb[fg] = (1 - fill_alpha) * rgb[fg] + fill_alpha * palette[labels[fg]]
        for i in range(1, int(labels.max()) + 1):
            rgb[_contours(labels == i)] = palette[i]
    if gt_masks is not None:
        for m in gt_masks:
            rgb[_contours(m)] = gt_color
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def summary_figure(scores: Sequence[Mapping], title: str = "", path=None):
    """The three rubric panels (IoU distribution, Dice distribution, 1:n / n:1
    relations) from the per-entry scores of ``src.pq_official.pq_score``
    (e.g. ``evaluate_frames(...)["per_entry"]`` or the OOF evaluation)."""
    from .pq_official import aggregate_scores

    plt = _plt()
    scored = [s for s in scores if s.get("scored", True)]
    agg = aggregate_scores(scored)
    ious = [v for s in scored for v in s.get("overlap_ious", [])]
    dices = [v for s in scored for v in s.get("overlap_dices", [])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.6))
    plot_overlap_distribution(ious, "IoU of overlapping pairs", "IoU", ax=axes[0])
    plot_overlap_distribution(dices, "Dice of overlapping pairs", "Dice", ax=axes[1])
    plot_relation_counts(agg["gt_degree_histogram"], agg["pred_degree_histogram"], ax=axes[2])
    fig.suptitle(title or f"micro PQ = {agg['pq']:.4f}  (SQ {agg['sq']:.3f} x RQ {agg['rq']:.3f}; "
                          f"TP {agg['tp']}, FP {agg['fp']}, FN {agg['fn']})")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
