# Final submission checklist

Competition final-material deadline announced by the organizers: **15 November 2026**.

## Reproducibility

- [ ] Repository is publicly accessible.
- [ ] `requirements.txt` contains exact versions.
- [ ] `.python-version` is recorded.
- [ ] `notebooks/filanet_end_to_end.ipynb` executes from the repository.
- [ ] Data layout is documented.
- [ ] Five final checkpoints are reproducible from the training command.
- [ ] Final checkpoint SHA-256 values are retained.
- [ ] Random seed and fold assignments are retained.
- [ ] Hardware/runtime are documented.

## Data integrity

- [ ] `scripts/audit_data.py --strict --decode-all` passes.
- [ ] 707 physical train observations are found.
- [ ] 1,154 independent train annotation records are found.
- [ ] 8,199 train filament instances are found.
- [ ] All images are 2048×2048.
- [ ] No physical observation crosses folds.
- [ ] No test labels or test-overlapping external annotations were used.

## Evaluation

- [ ] Local PQ semantics have been checked against the latest organizer notebook.
- [ ] Strict `IoU > 0.50` edge case has a regression test.
- [ ] OOF macro PQ is recorded.
- [ ] OOF micro PQ is recorded.
- [ ] Per-fold PQ is recorded.
- [ ] Bootstrap interval is recorded.
- [ ] Dice and IoU distributions are recorded.
- [ ] One-to-many / many-to-one diagnostics are recorded.
- [ ] Ablation results are recorded.
- [ ] Post-processing nested-selection estimate is recorded.

## Submission artifact

- [ ] Five-fold ensemble is generated from the frozen tuned config.
- [ ] CSV contains exactly `filament_id,segmentation_rle`.
- [ ] IDs are unique.
- [ ] Every RLE decodes at the correct image size.
- [ ] No instance is empty.
- [ ] Predicted instances do not overlap.
- [ ] `submission.verification.json` says `"ok": true`.
- [ ] Submission SHA-256 is retained.

## Four-page technical report

- [ ] Uses the organizer-provided report template.
- [ ] Pipeline figure.
- [ ] Preprocessing and supervision derivation.
- [ ] Architecture and losses.
- [ ] Instance reconstruction.
- [ ] Leakage-safe validation protocol.
- [ ] Main OOF result with uncertainty.
- [ ] Ablation table.
- [ ] IoU/Dice and fragmentation diagnostics.
- [ ] Qualitative examples and failure cases.
- [ ] Runtime/compute discussion.
- [ ] Limitations and annotator disagreement.
- [ ] Every numerical claim comes from a saved artifact.
