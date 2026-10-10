# Experiment protocol

Goal: keep only the changes that improve the **official micro PQ** under
leakage-safe validation, and be able to show it.

## Splits

`cv.group_mode: day` (default) keeps every observation of a UT day — all
GONG sites, all annotators — in one fold. `observation` only groups the
annotators of one image; `month` is a stricter stress test. No OOF prediction
may come from a model that saw the same day.

## Model selection inside a fold

`scripts/train.py` scores the held-out fold with the official micro PQ every
`train.val_interval` epochs (default 2; `val_max_images` subsamples for speed)
and keeps the best epoch as `fold{k}_best.pt`. The training loss is not used
for selection.

## Comparing two variants

1. Train both on the same folds, seed, encoder initialisation and epochs.
2. Cache OOF predictions and evaluate both with the same post-processing.
3. `scripts/compare_oof_experiments.py --baseline A/records.csv --candidate B/records.csv`
   — paired bootstrap of the micro PQ over observations.
4. Keep the change only if the 95 % interval of the difference excludes zero
   or, at least, the probability of improvement is high and the rubric
   diagnostics (fragmentation, merging, misses, spurious) do not degrade.

## Campaigns

| script | content | training needed |
|---|---|---|
| `run_ablation.py` | baseline → +topology (clDice) → +orientation → +graph bridge → +TTA → +full-resolution refiner | baseline, topology, orientation, full-res |
| `run_innovation_ablation.py` | V2: limb gate, multi-scale context, frequency branch, width, curvature, endpoint, junction, centerline CE, radius balance, uncertainty | every step |
| `run_v3_ablation.py` | V2 reference, H0 loss only, embedding only, geodesic repair only, all V3 | all but geodesic-only |
| `run_supervised_campaign.py` | ImageNet vs ImageNet → GONG self-supervised encoder | both |

Each script writes a manifest (dry run by default) with the exact commands,
reuses checkpoints/OOF caches for inference-only steps and appends the paired
comparison of consecutive variants. Screen on one fold with fewer epochs
(`--folds 0 --epochs 8`), then confirm on all folds.

## Post-processing

`scripts/optimize_postprocess.py` (coordinate ascent by default) tunes the
reconstruction on the cached OOF maps with a nested estimate (see
`docs/evaluation_protocol.md`). It never sees the test set or the leaderboard.

## Experiment ledger (for the report)

For every variant keep: git commit, config, fold checkpoints + SHA-256, per-fold
and OOF micro PQ with CI, SQ/RQ, TP/FP/FN, IoU/Dice distributions,
fragmentation/merge rates, runtime and hardware, the decision and a one-line
reason. Negative results go in the report too: they show that the choices were
measured.
