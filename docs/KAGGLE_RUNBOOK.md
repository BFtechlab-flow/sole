# Running FILA-Net on Kaggle (official data, GPU)

Everything runs from one notebook, `notebooks/filanet_end_to_end.ipynb`; only
its settings cell changes between sessions. Kaggle stops a session after
12 hours and the weekly GPU quota is limited (see *Settings → Quotas* on
Kaggle), so the work is split into a few sessions that resume each other.

## Timing (estimates — measure your own)

Measured on a CPU here: building one training sample (2048 × 2048 image →
4 channels + targets of a 1024 px patch) takes ~0.85 s on one core;
reconstructing the instances of one image takes ~0.45 s. GPU speed was not
measured (no GPU and no competition data were available while writing this).
After the first epoch, read `epoch_seconds` and `wall_seconds` in
`checkpoints/fold0_log.jsonl` and plan with those numbers.

Rough expectation on a Kaggle T4: 3–5 h per fold for 30 epochs including
validation, i.e. 2–3 folds per session; OOF caching + calibration + test
inference ≈ 4 h (`POST_TRAINING_HOURS`).

## One-time setup

1. *Create → New Notebook*, then *File → Import Notebook* with
   `notebooks/filanet_end_to_end.ipynb` (uploaded from your computer or from
   its GitHub URL). The notebook clones this repository into
   `/kaggle/working/sole` when it runs outside it.
2. *Add Input → Competition*: **Solar Filament Segmentation Challenge 2026**.
3. *Settings*: Accelerator **GPU T4** (or P100), **Internet on** (needed for
   `git clone`, `pip` and the ImageNet weights downloaded by `timm`).
4. Run once with `MODE = "smoke"` (about 2 minutes) to check the environment.

## Session plan

Long runs use *Save Version → Save & Run All (Commit)*: they keep running when
the browser is closed and their `/kaggle/working` folder becomes the version's
output (checkpoints, logs, figures, metrics).

| session | settings cell | produces |
|---|---|---|
| A | `MODE = "full"`, `FOLDS = [0, 1, 2]` | `fold0..2_best.pt`, `fold*_last.pt`, logs |
| B | add **A's output** as input, `RESUME_FROM = "/kaggle/input/<A's output>/sole/checkpoints"` (check the exact path in the input panel), `FOLDS = [0, 1, 2, 3, 4]` | remaining folds (finished folds are skipped instantly) |
| C (if needed) | same as B with B's output | the rest of the folds, then steps 5–10 |

Steps 5–10 (OOF predictions, official OOF evaluation, nested calibration,
ablation plan, `submission.csv`, figures, weights manifest) start
automatically in the first session where every fold is finished **and** at
least `POST_TRAINING_HOURS` remain; otherwise the notebook says so and the next
session (with `RESUME_FROM`) does them without retraining.

The calibration searches on `POSTPROCESS_IMAGES_PER_FOLD = 60` images per fold
by default (CPU-bound); the tuned setting is then evaluated on **all** OOF
images (`artifacts/oof_metrics_tuned.json`). Set it to `None` in a session
with spare time for a full-data search.

## Before the first submission

* Read `artifacts/data_audit.json`: the counts must match the announced layout
  (707 / 1,154 / 8,199, all 2048 × 2048); note the **human baseline**
  (`inter_annotator_official_pq`) for the report.
* `artifacts/oof_metrics.json` and `oof_metrics_tuned.json`: official micro PQ
  with a 95 % bootstrap interval; `postprocess_search.json`: the **nested**
  estimate (the honest number for the report).
* `artifacts/submission.verification.json` must say `"ok": true`.

## Submitting

* *Competition → Submit Prediction*, choose `submission.csv` from the notebook
  output (5 submissions per day; select up to 2 as final).
* Optional check with the organizers' own code: download their
  *Self_Evaluation_Notebook* and run
  `python tools/verify_against_official_notebook.py <notebook>.ipynb --gt artifacts/oof_export/oof_gt.csv --pred artifacts/oof_export/oof_pred.csv`.

## Final materials (Google form, by 15 November 2026)

1. **Weights**: download `checkpoints/fold*_best.pt` from the output, create
   the GitHub release `v1.0` with the five files, commit
   `weights/manifest.json` and `configs/filanet_tuned.yaml` (both written by
   the notebook in full mode). Check: `python tools/weights.py fetch`.
2. **Repository**: public, `requirements.txt` pinned, notebook present.
3. **Report**: fill `report/report_draft.tex` into the organizers' Overleaf
   template with the numbers of the artifacts above, export the PDF (4 pages
   excluding acknowledgment and references).

## Ablations within the quota

Training every ablation variant on five folds is not affordable on a free
quota. Screen on one fold with fewer epochs, then confirm only the promising
changes on all folds:

```bash
python scripts/run_ablation.py --folds 0 --epochs 8 --execute
python scripts/run_v3_ablation.py --config configs/filanet_v3.yaml --folds 0 --epochs 8 --execute
```

Inference-only variants (graph bridge, TTA, geodesic repair) reuse the
checkpoints of the previous variant and cost no training. Report every result,
including the negative ones.
