"""Draw the pipeline figure of the technical report (report/figures/pipeline.{pdf,png}).

The figure is generated from code so that it stays in sync with the
repository; it contains no result, only the structure of the method.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK = "#1f2933"
MUTED = "#52606d"
FILL = {"data": "#e3f2fd", "net": "#fff3e0", "post": "#e8f5e9", "eval": "#f3e5f5"}
EDGE = {"data": "#1565c0", "net": "#ef6c00", "post": "#2e7d32", "eval": "#6a1b9a"}


def box(ax, x, y, w, h, title, lines, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                linewidth=1.0, edgecolor=EDGE[kind], facecolor=FILL[kind]))
    ax.text(x + w / 2, y + h - 0.04, title, ha="center", va="top", fontsize=6.6, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.13, "\n".join(lines), ha="center", va="top", fontsize=5.5, color=MUTED,
            linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, text=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, linewidth=0.9, color=INK))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.02, text, ha="center", va="bottom", fontsize=5.4, color=MUTED)


def draw(out_dir: Path):
    fig = plt.figure(figsize=(7.0, 2.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(-0.01, 1.0)
    ax.axis("off")

    top, h, w, gap = 0.53, 0.43, 0.117, 0.0265
    xs = [0.008 + i * (w + gap) for i in range(7)]
    stages = [
        ("H-alpha image", ["GONG JPEG", "2048 x 2048, 8-bit", "grayscale"], "data"),
        ("4 input channels", ["robust normalisation", "CLAHE", "ridge (DoG)", "radial position"], "data"),
        ("ConvNeXt-T+FPN", ["ImageNet init.", "stride-4 decoder", "context / frequency", "limb gate (opt.)"], "net"),
        ("dense fields", ["region, centerline", "boundary, distance", "axial orientation", "width, keypoints,", "embedding (opt.)"], "net"),
        ("tiled inference", ["1024 px tiles", "256 px overlap", "Hann blending", "flip TTA, AMP", "5-fold ensemble"], "net"),
        ("instances", ["threshold, seeds", "endpoint graph", "repair, marker", "watershed, fill,", "dilate, filter"], "post"),
        ("submission", ["disjoint masks", "COCO RLE counts", "validated CSV"], "post"),
    ]
    for x, (title, lines, kind) in zip(xs, stages):
        box(ax, x, top, w, h, title, lines, kind)
    for i in range(6):
        arrow(ax, xs[i] + w, top + h / 2, xs[i + 1], top + h / 2)

    bot, hb = 0.04, 0.32
    wb = (1 - 0.016 - 3 * 0.04) / 4
    xb = [0.008 + i * (wb + 0.04) for i in range(4)]
    bottom = [
        ("supervision", ["targets derived from the", "polygons only; BCE + Dice,", "clDice, orientation, ..."], "eval"),
        ("5-fold CV by UT day", ["best epoch = best official", "micro PQ on held-out fold", "(every annotator scored)"], "eval"),
        ("out-of-fold maps", ["one prediction per image", "official PQ + bootstrap CI", "rubric distributions"], "eval"),
        ("nested calibration", ["post-processing chosen on", "other folds, scored on the", "held-out fold; ablations"], "eval"),
    ]
    for x, (title, lines, kind) in zip(xb, bottom):
        box(ax, x, bot, wb, hb, title, lines, kind)
    for i in range(3):
        arrow(ax, xb[i] + wb, bot + hb / 2, xb[i + 1], bot + hb / 2)
    arrow(ax, xb[0] + wb / 2, bot + hb, xs[2] + w / 2, top, None)
    ax.text((xb[0] + wb / 2 + xs[2] + w / 2) / 2 - 0.02, (bot + hb + top) / 2, "trains", fontsize=5.4, color=MUTED,
            ha="right", va="center")
    arrow(ax, xb[3] + wb / 2, bot + hb, xs[5] + w / 2, top, None)
    ax.text((xb[3] + wb / 2 + xs[5] + w / 2) / 2 + 0.015, (bot + hb + top) / 2, "tunes", fontsize=5.4, color=MUTED,
            ha="left", va="center")

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"pipeline.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    return out_dir / "pipeline.pdf"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="report/figures")
    print(draw(Path(ap.parse_args().out_dir)))


if __name__ == "__main__":
    main()
