"""Test-set inference -> submission.csv (competition format) + verification report.

Only the supplied test images are used (no other observation at inference
time, as required by the organizers). The fold ensemble's maps go through the
same quantisation as the OOF cache, then through the tuned post-processing.
The report records the end-to-end runtime (images/second), which is part of
the judging rubric.

``--maps-dir`` stores the quantised ensemble maps of every test image (and
reuses the ones already there): the post-processing can then be re-run with
another config (e.g. the tuned one) on a CPU, without the networks, and gives
exactly the same instances as a fresh prediction would.
"""

import os
import sys
from pathlib import Path as _Path

# less CUDA memory fragmentation (must be set before torch initialises CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `src` imports work

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import pandas as pd
import torch

from src.engine import load_models, postprocess_params, predict_label_map
from src.evaluation import label_map_to_rles
from src.oof import RECONSTRUCTION_KEYS, available_prediction, load_prediction, save_prediction
from src.reconstruct import reconstruct_label_map
from src.submission import save_verification, validate_submission_file, write_submission
from src.utils import data_paths, image_files, load_yaml


def write_maps_manifest(maps_dir, weights, cfg):
    """Which checkpoints produced the cached maps (checked before they are reused)."""
    import hashlib

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    inf = cfg["inference"]
    manifest = {"weights_sha256": [sha(w) for w in weights],
                "prediction": {k: inf.get(k) for k in ("tile", "overlap", "tta", "amp")}}
    maps_dir.mkdir(parents=True, exist_ok=True)
    path = maps_dir / "manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old != manifest:
            raise SystemExit(f"{maps_dir} holds maps of other checkpoints/settings: use another --maps-dir")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--weights", nargs="+", default=None,
                    help="fold checkpoints (not needed when --maps-dir already holds every test image)")
    ap.add_argument("--maps-dir", default=None, help="cache of the quantised ensemble maps (read and written)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly before this budget (maps written so far are kept; rerun to resume)")
    ap.add_argument("--output", default="submission.csv")
    ap.add_argument("--test-images", default=None, help="override the configured test image folder")
    ap.add_argument("--max-images", type=int, default=None, help="debug: only the first N images")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_dir = Path(args.test_images) if args.test_images else data_paths(cfg)[3]
    paths = image_files(test_dir)
    if args.max_images:
        paths = paths[: args.max_images]
    if not paths:
        raise FileNotFoundError(f"no test images in {test_dir}")
    maps_dir = Path(args.maps_dir) if args.maps_dir else None
    cached = {p.stem for p in paths if maps_dir is not None and available_prediction(maps_dir / f"{p.stem}.npz")}

    models, t_load = [], 0.0
    if len(cached) < len(paths):
        if not args.weights:
            raise SystemExit(f"{len(paths) - len(cached)} test image(s) have no cached maps: --weights is required")
        t_load = time.time()
        models = load_models(cfg, args.weights, device)
        t_load = time.time() - t_load
        if maps_dir is not None:
            write_maps_manifest(maps_dir, args.weights, cfg)

    deadline = time.time() + 3600.0 * args.max_hours if args.max_hours else None
    rows, image_shapes, predicted = [], {}, 0
    t0 = time.time()
    for pos, path in enumerate(paths, 1):
        if deadline is not None and path.stem not in cached and time.time() > deadline:
            print(f"time budget used after {pos - 1}/{len(paths)} images: maps kept in {maps_dir}; "
                  f"rerun to continue (no submission written)", flush=True)
            return
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(path)
        image_shapes[path.stem] = gray.shape
        if path.stem in cached:
            pred = load_prediction(maps_dir / f"{path.stem}.npz", keys=RECONSTRUCTION_KEYS)
            labels, _ = reconstruct_label_map(
                pred["region"], pred["centerline"], pred["boundary"], pred["distance"],
                orientation=pred.get("orientation"), width=pred.get("width"),
                instance_embedding=pred.get("instance_embedding"), **postprocess_params(cfg),
            )
        else:
            labels, _, pred = predict_label_map(models, gray, cfg, device)
            predicted += 1
            if maps_dir is not None:
                save_prediction(maps_dir / f"{path.stem}.npz", {k: pred[k] for k in RECONSTRUCTION_KEYS if k in pred})
        for k, rle in enumerate(label_map_to_rles(labels), 1):
            rows.append({"filament_id": f"{path.stem}_{k}", "segmentation_rle": rle["counts"].decode("ascii")})
        if pos % 10 == 0 or pos == len(paths):
            print(f"inference {pos}/{len(paths)} images; rows={len(rows)}", flush=True)
    seconds = time.time() - t0

    frame = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    write_submission(frame, args.output)
    print("saved", args.output, len(rows))

    runtime = {
        "images": len(paths),
        "images_predicted_by_the_networks": predicted,
        "images_post_processed_from_cached_maps": len(paths) - predicted,
        "seconds": round(seconds, 2),
        "seconds_per_image": round(seconds / len(paths), 3),
        "images_per_second": round(len(paths) / seconds, 4),
        "model_load_seconds": round(t_load, 2),
        "models_in_ensemble": len(models) if models else None,
        "tta": bool(cfg["inference"].get("tta", True)),
        "amp": bool(cfg["inference"].get("amp", False)) and device == "cuda",
        "device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or "cpu",
        "torch": torch.__version__,
    }
    if predicted < len(paths):
        runtime["note"] = ("end-to-end speed is only measured when every image is predicted by the networks; "
                           "see the verification report of that run")
    report = {"ok": True, "runtime": runtime}
    if not args.skip_validation:
        expected = tuple(cfg.get("submission", {}).get("expected_shape", [2048, 2048]))
        report = validate_submission_file(args.output, image_shapes, require_nonoverlap=True, expected_shape=expected)
        report["runtime"] = runtime
    report = save_verification(report, args.output, str(Path(args.output).with_suffix(".verification.json")))
    print(json.dumps({k: v for k, v in report.items() if k not in ("warnings",)}, indent=2))
    if not report["ok"]:
        raise SystemExit("submission validation failed")


if __name__ == "__main__":
    main()
