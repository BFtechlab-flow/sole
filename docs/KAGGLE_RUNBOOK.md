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

Measured in the first full session (Kaggle, one Tesla T4, `configs/filanet.yaml`):
a training epoch takes ~500 s and a validation pass (139 held-out
observations, official PQ) ~480 s every second epoch, so one fold (30 epochs)
takes ~6.2 h. With *GPU T4 × 2* the notebook trains two folds at once, one
per GPU. In the second session (two folds in parallel, 4 vCPUs) epochs took
820–1080 s and validation passes ~760 s: the CPUs, not the GPUs, were the
limit (only ~10 % more throughput than one GPU). Building a training sample
was then made 1.8× faster with bit-identical results (channels computed on
the patch, per-worker caches; `tests/test_channels_window.py`). OOF caching + calibration
+ test inference: see the session plan below (measured in the third session).

## One-time setup

1. *Create → New Notebook*, then *File → Import Notebook* with
   `notebooks/filanet_end_to_end.ipynb` (uploaded from your computer or from
   its GitHub URL). The notebook clones this repository into
   `/kaggle/working/sole` when it runs outside it.
2. *Add Input → Competition*: **Solar Filament Segmentation Challenge 2026**.
3. *Settings*: Accelerator **GPU T4 × 2** (or one T4 / P100), **Internet on** (needed for
   `git clone`, `pip` and the ImageNet weights downloaded by `timm`).
4. Run once with `MODE = "smoke"` (about 2 minutes) to check the environment.

## Session plan

Long runs use *Save Version → Save & Run All (Commit)*: they keep running when
the browser is closed and their `/kaggle/working` folder becomes the version's
output (checkpoints, logs, figures, metrics).

| session | settings cell | produces |
|---|---|---|
| A | `MODE = "full"` | as many epochs/folds as fit in `SESSION_HOURS`: `fold*_best.pt`, `fold*_last.pt`, `fold*_log.jsonl` |
| B, C, … | nothing to change: *Add Input → Your Work → Notebooks* → the previous version of this notebook | the notebook finds the previous checkpoints in `/kaggle/input`, restores the most advanced copy of every fold, skips finished folds and resumes the others from `fold*_last.pt` |

The first cell of section 4 prints the progress of every fold
(`fold k: e/30 epochs`). `RESUME_FROM` is only needed to force a specific
folder.

After training, the remaining steps are split by the resource they need and
every one of them stops cleanly before the end of the session and resumes in
the next one (never rely on Kaggle's 12 h limit: a run killed by it may lose
its output).

| session | accelerator | what runs | measured / expected time |
|---|---|---|---|
| D | **GPU T4 × 2** | OOF cache of the 707 training images (one fold model each) on one GPU, test maps of the 180 test images (5-fold ensemble, TTA) on the other, `submission_default.csv`, official OOF evaluation | OOF cache 66 min on one T4 (measured), test maps ≈ 1.5 h (estimate) |
| E, … | **None** (CPU only, no GPU quota) | nested post-processing calibration (`postprocess_memo.npz` keeps every evaluated setting), then the calibrated `submission.csv` from the cached test maps | ≈ 100 s per candidate setting on 300 OOF images with 4 CPUs (measured); a full nested search is several hundred candidates, i.e. one or two CPU sessions |

The OOF cache (~5 GB), the test maps and the calibration memo are saved in the
output (`artifacts/`) and restored from the attached previous output only when
they were produced by exactly the current checkpoints (SHA-256 check).
`CALIBRATE_WITH_GPU = True` runs the calibration in a GPU session too (it
does not use the GPU).

Kaggle's preinstalled packages (and its CUDA build of `torch`) are kept; only
packages of `requirements.txt` that are missing are installed, and the versions
actually used are saved in `artifacts/environment.txt`.

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
