# Final submission checklist (deadline: 15 November 2026)

Final materials go through the organizers' Google form (Overview → *Final
Submission*): the report PDF and the public repository link.

## Repository

- [ ] Public, readable without any permission request; stays public at least
      until the winners are announced (30 Nov 2026).
- [ ] `requirements.txt` lists every package with its exact version.
- [ ] `notebooks/filanet_end_to_end.ipynb` illustrates and runs the entire
      pipeline (`tests/test_notebook_smoke.py` passes).
- [ ] `weights/manifest.json` lists the five checkpoints of the submission
      with SHA-256; the GitHub release with the files exists;
      `python tools/weights.py fetch` works on a clean clone.
- [ ] `configs/filanet_tuned.yaml` (the exact post-processing of the
      submission) is committed.
- [ ] `python -m pytest -q` passes.

## Data and rules

- [ ] `scripts/audit_data.py` matches the announced layout (707 observations,
      1,154 records, 8,199 filaments, 2048 × 2048).
- [ ] Only the competition annotations were used; the public MAGFiLO release
      was never downloaded.
- [ ] Inference uses only the test images (no other observation).
- [ ] No hand labelling or manual correction of test predictions.

## Numbers for the report (each from a saved artifact)

- [ ] Human baseline: `data_audit.json → inter_annotator_official_pq`.
- [ ] OOF micro PQ + 95 % CI, SQ, RQ, TP/FP/FN, per-fold PQ: `oof_metrics.json`.
- [ ] Nested calibration estimate and tuned OOF PQ: `postprocess_search.json`,
      `oof_metrics_tuned.json`.
- [ ] IoU/Dice distributions, 1:n / n:1 relations: `oof_export*/oof_rubric_panels.png`.
- [ ] Ablation table with paired deltas and CIs: `ablations/compare_*.json`.
- [ ] Runtime (images/s, GPU, ensemble size, TTA, AMP): `submission.verification.json`.
- [ ] Public leaderboard score (only as a sanity check, never for tuning).

## Kaggle submission

- [ ] `submission.verification.json` says `"ok": true`.
- [ ] Up to two final submissions selected on Kaggle.

## Report (organizers' Overleaf template)

- [ ] Black template text unchanged; gray guide text replaced;
      `\guideStylefalse` set; at most 4 pages excluding acknowledgment and
      references.
- [ ] Methodology: algorithm, end-to-end pipeline (pre/post-processing),
      sampling and splits, pipeline figure, example segmentations.
- [ ] Evaluation: strategy, tables/plots with uncertainty, strengths,
      weaknesses and possible improvements.
- [ ] Every number traceable to an artifact; no unmeasured claim.
