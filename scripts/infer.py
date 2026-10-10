"""Test-set inference -> submission.csv (competition format) + verification report.

Only the supplied test images are used (no other observation at inference
time, as required by the organizers). The fold ensemble's maps go through the
same quantisation as the OOF cache, then through the tuned post-processing.
The report records the end-to-end runtime (images/second), which is part of
the judging rubric.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `src` imports work

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import pandas as pd
import torch

from src.engine import load_models, predict_label_map
from src.evaluation import label_map_to_rles
from src.submission import save_verification, validate_submission_file, write_submission
from src.utils import data_paths, image_files, load_yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--output", default="submission.csv")
    ap.add_argument("--test-images", default=None, help="override the configured test image folder")
    ap.add_argument("--max-images", type=int, default=None, help="debug: only the first N images")
    ap.add_argument("--skip-validation", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_load = time.time()
    models = load_models(cfg, args.weights, device)
    t_load = time.time() - t_load

    test_dir = Path(args.test_images) if args.test_images else data_paths(cfg)[3]
    paths = image_files(test_dir)
    if args.max_images:
        paths = paths[: args.max_images]
    if not paths:
        raise FileNotFoundError(f"no test images in {test_dir}")

    rows, image_shapes = [], {}
    t0 = time.time()
    for pos, path in enumerate(paths, 1):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(path)
        image_shapes[path.stem] = gray.shape
        labels, _, _ = predict_label_map(models, gray, cfg, device)
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
        "inference_seconds": round(seconds, 2),
        "seconds_per_image": round(seconds / len(paths), 3),
        "images_per_second": round(len(paths) / seconds, 4),
        "model_load_seconds": round(t_load, 2),
        "models_in_ensemble": len(models),
        "tta": bool(cfg["inference"].get("tta", True)),
        "amp": bool(cfg["inference"].get("amp", False)) and device == "cuda",
        "device": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or "cpu",
        "torch": torch.__version__,
    }
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
