"""Cross-check ``src.pq_official`` against the organizers' own notebook code.

The organizers' *Self_Evaluation_Notebook* is not redistributed in this
repository. To run the check:

1. on Kaggle open https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
   and download it (``... > Download``), e.g. as ``self-evaluation-notebook.ipynb``;
2. run (needs torch, pandas, pycocotools; matplotlib is imported by the notebook)::

       python tools/verify_against_official_notebook.py self-evaluation-notebook.ipynb
       # or on real out-of-fold files written by scripts/evaluate_oof.py --export-dir:
       python tools/verify_against_official_notebook.py self-evaluation-notebook.ipynb \\
           --gt artifacts/oof_export/oof_gt.csv --pred artifacts/oof_export/oof_pred.csv

Only the notebook cells that import modules or define functions are executed
(the cells that load data, compute or plot are skipped); the script then scores
the same inputs with the notebook's ``get_overlap_df`` + ``get_pq_score`` and
with ``src.pq_official.evaluate_frames`` and prints both values. Without
``--gt/--pred`` it generates random 2048 x 2048 multi-annotator scenes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pq_official import evaluate_frames  # noqa: E402
from src.rle import encode_counts  # noqa: E402

REQUIRED = ("get_overlap_df", "get_pq_score")


def load_notebook_functions(ipynb: Path) -> dict:
    nb = json.loads(ipynb.read_text(encoding="utf-8"))
    namespace: dict = {"__name__": "organizers_notebook"}
    executed = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        body = [ln for ln in src.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
        if not body:
            continue
        only_imports = all(ln.startswith(("import ", "from ")) for ln in body)
        defines = any(ln.startswith("def ") for ln in body)
        if not (only_imports or defines):
            continue
        exec(compile(src, f"<{ipynb.name}>", "exec"), namespace)  # noqa: S102 - user-supplied official notebook
        executed += 1
    missing = [f for f in REQUIRED if f not in namespace]
    if missing:
        raise RuntimeError(f"{ipynb} does not define {missing}; is it the organizers' notebook?")
    print(f"executed {executed} import/definition cells of {ipynb.name}")
    return namespace


def random_case(seed: int, size: int = 2048) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    shape = (size, size)
    gt_rows, pred_rows = [], []
    for i in range(int(rng.integers(2, 5))):
        stem = f"2015012517{i:04d}Mh"
        base = []
        for _ in range(int(rng.integers(1, 5))):
            m = np.zeros(shape, bool)
            y, x = rng.integers(200, size - 400, size=2)
            m[y : y + int(rng.integers(8, 40)), x : x + int(rng.integers(60, 300))] = True
            base.append(m)
        for a in range(int(rng.integers(1, 4))):
            kept = [ndi.binary_dilation(m, iterations=int(rng.integers(1, 4))) if rng.random() < 0.5 else m
                    for m in base if rng.random() > 0.2] or [base[0]]
            gt_rows += [(f"0104{a:02d}-{stem}_{k}", encode_counts(m)) for k, m in enumerate(kept, 1)]
        taken = np.zeros(shape, bool)
        k = 0
        for m in base:
            r = rng.random()
            if r < 0.15:
                continue
            p = np.roll(m, tuple(int(v) for v in rng.integers(-6, 7, size=2)), axis=(0, 1)) if r < 0.6 \
                else ndi.binary_dilation(m, iterations=5)
            p &= ~taken
            taken |= p
            if p.any():
                k += 1
                pred_rows.append((f"{stem}_{k}", encode_counts(p)))
    cols = ["filament_id", "segmentation_rle"]
    return pd.DataFrame(gt_rows, columns=cols), pd.DataFrame(pred_rows, columns=cols)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", type=Path, help="the organizers' Self_Evaluation_Notebook (.ipynb)")
    ap.add_argument("--gt", help="ground-truth CSV (filament_id, segmentation_rle)")
    ap.add_argument("--pred", help="prediction CSV in submission format")
    ap.add_argument("--cases", type=int, default=3, help="random scenes when --gt/--pred are not given")
    args = ap.parse_args()

    official = load_notebook_functions(args.notebook)
    if args.gt and args.pred:
        cases = [(pd.read_csv(args.gt, dtype=str, keep_default_na=False),
                  pd.read_csv(args.pred, dtype=str, keep_default_na=False))]
    else:
        cases = [random_case(s) for s in range(args.cases)]

    worst = 0.0
    for i, (gt_df, pred_df) in enumerate(cases):
        pq_official = float(official["get_pq_score"](official["get_overlap_df"](gt_df, pred_df)))
        pq_ours = float(evaluate_frames(gt_df, pred_df)["pq"])
        diff = abs(pq_official - pq_ours)
        worst = max(worst, diff)
        print(f"case {i}: organizers' notebook PQ = {pq_official:.8f} | src.pq_official = {pq_ours:.8f} | diff = {diff:.1e}")
    ok = worst < 1e-6
    print("MATCH" if ok else "MISMATCH", f"(max |diff| = {worst:.1e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
