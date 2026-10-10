# Evaluation protocol

## Reference

* Competition overview, *Evaluation* and *Leaderboard Ranking* sections
  (updated 7 Aug 2026): <https://www.kaggle.com/competitions/filament-segmentation-2026/overview>
* Organizers' *Self_Evaluation_Notebook* (v6, 9 Aug 2026), described by the
  host as the exact implementation of the leaderboard PQ:
  <https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook>
* Panoptic Quality: Kirillov et al., *Panoptic Segmentation*, CVPR 2019
  (<https://arxiv.org/abs/1801.00868>).

## Official score (what `src/pq_official.py` computes)

1. The ground truth is split into **annotator-image entries**: one label set
   of one observation (`<annotator batch>-<stem>`). An observation labelled by
   three experts gives three entries, each compared with the same prediction
   set of that image.
2. Inside an entry, a (GT, prediction) pair is a **hit when IoU > 0.5**
   (strictly). Every hit is a TP and contributes its IoU; a prediction without
   hit is an FP; a GT filament without hit is an FN. There is no one-to-one
   assignment step (with disjoint predictions — which the pipeline always
   produces — hits are unique anyway).
3. TP, FP, FN and the sum of matched IoUs are accumulated over **all**
   entries, then `PQ = ΣIoU / (TP + ½FP + ½FN)` is computed once: a **micro**
   average. The mean of per-entry PQ ("macro") is reported only as a
   diagnostic (`macro_pq_diagnostic`); it never drives a decision.
4. Only images present in the ground truth are scored.

IoUs are computed on COCO RLEs with pycocotools and re-derived from the exact
integer intersection/union in float32, i.e. the same arithmetic as the
notebook's dense torch code.

### Verification

* `tests/test_official_parity.py` — random multi-annotator scenes (missed,
  split, merged, shifted, dilated, spurious, duplicated filaments, empty
  prediction sets, predictions on unscored images) scored by
  `src/pq_official.py` and by an independent dense re-implementation
  (`tests/reference_pq_dense.py`): PQ and TP/FP/FN must agree.
* `tools/verify_against_official_notebook.py` — runs the organizers' own
  notebook functions (download the notebook from Kaggle; it is not
  redistributed here) on random 2048 × 2048 scenes or on real OOF exports and
  prints both scores. Checked on 10 Oct 2026 with the code cells of notebook
  version 6 as published on Kaggle: identical PQ (|Δ| = 0) on every random
  scene. Re-run it on the real OOF export once it exists.

## Rubric diagnostics (judging, 70 % quantitative)

Written by `scripts/evaluate_oof.py` (JSON) and drawn by `src/plots.py`:

* distribution of IoU and of Dice over every overlapping (GT, prediction) pair
  (`overlap_*_distribution`) and over the matched pairs (`matched_*`);
* one-to-many / many-to-one relations with the notebook's convention
  (overlap = IoU > 0): `gt_degree_histogram` (0 = missed, ≥ 2 = fragmented)
  and `pred_degree_histogram` (0 = spurious, ≥ 2 = merging), with the
  `fragmentation_rate`, `merge_rate`, `miss_rate` and `spurious_rate`;
* end-to-end runtime of the test inference (`submission.verification.json`,
  images per second, device, ensemble size, TTA/AMP flags).

## Uncertainty

* `bootstrap_micro_pq` resamples **whole observations** (their 1–3 entries
  are not independent) and recomputes the micro PQ from summed counts;
  2,000 draws, 95 % percentile interval.
* `paired_bootstrap_micro_pq` (used by `scripts/compare_oof_experiments.py`)
  resamples the same observations for two experiments and reports the
  distribution of `PQ(candidate) − PQ(baseline)`.

## Leakage control

* Folds group all observations of the same UT day (`cv.group_mode: day`):
  GONG sites observe the same Sun minutes apart and filaments persist for
  days, so same-day images in different folds would inflate the OOF score.
* Post-processing is calibrated with a **nested** procedure: for each outer
  fold the setting is chosen on the other folds and scored on the held-out
  one; the resulting `nested_micro_pq` is the number to report. The final
  setting (chosen on all folds) is used only for the test set.
* The leaderboard is never used to choose anything.
