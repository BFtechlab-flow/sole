"""Standalone gate for a submission CSV (run before every upload)."""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `src` imports work

import argparse
import json
from pathlib import Path

import cv2

from src.submission import save_verification, validate_submission_file
from src.utils import image_files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("submission")
    ap.add_argument("--test-images", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--allow-overlap", action="store_true")
    args = ap.parse_args()

    image_shapes = {}
    for path in image_files(args.test_images):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        image_shapes[path.stem] = image.shape

    report = validate_submission_file(args.submission, image_shapes, require_nonoverlap=not args.allow_overlap)
    output = args.output or str(Path(args.submission).with_suffix(".verification.json"))
    report = save_verification(report, args.submission, output)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
