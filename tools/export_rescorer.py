"""Convert a ``rescorer.joblib`` into the version-independent ``rescorer.json``.

A scikit-learn pickle loads only with the scikit-learn version that wrote it
(the final run used Kaggle's scikit-learn 1.6.1). Run this once with that
version; the JSON is then read by ``src.rescore.load_rescorer`` with plain
numpy, giving the same predictions:

    pip install scikit-learn==1.6.1
    python tools/export_rescorer.py artifacts/rescoring/rescorer.joblib weights/rescorer.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rescore import TreeEnsemble, export_rescorer  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("joblib_path")
    ap.add_argument("json_path")
    args = ap.parse_args(argv)
    import joblib

    bundle = joblib.load(args.joblib_path)
    out = export_rescorer(bundle, args.json_path)
    # check: identical predictions on random inputs spanning several orders of magnitude, with missing values
    rng = np.random.default_rng(0)
    n = len(bundle["feature_names"])
    X = rng.normal(size=(20000, n)) * rng.uniform(0.01, 1000, size=n)
    X[rng.random(X.shape) < 0.01] = np.nan
    ref = bundle["model"].predict(X)
    got = TreeEnsemble.from_dict(__import__("json").loads(Path(out).read_text())["model"]).predict(X)
    diff = float(np.max(np.abs(ref - got)))
    print(f"wrote {out} ({Path(out).stat().st_size} bytes); max |difference| on 20000 random rows: {diff:.3g}")
    return 0 if diff <= 1e-12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
