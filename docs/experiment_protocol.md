# Experiment protocol

The objective is not to accumulate features. It is to retain only changes that
transfer under leakage-safe validation.

## Split discipline

Default `cv.group_mode: observation` keeps all annotator records of a physical
GONG JPEG in one fold. `day` and `month` modes are available for stricter
temporal stress tests.

No prediction used for OOF scoring may come from a model trained on that
physical image.

## Primary campaign

Run:

```bash
python scripts/run_ablation.py
```

Then execute the generated manifest on a GPU machine.

The required ordered comparison is:

1. baseline;
2. + topology/clDice;
3. + orientation supervision;
4. + graph bridging;
5. + TTA.

A capability should be retained only when the improvement is larger than normal
fold variation and does not materially damage fragmentation/merge diagnostics.

## Post-processing tuning

`optimize_postprocess.py` does not select parameters on the fold it reports.
For each held-out outer fold:

- candidate quality is estimated from all other folds;
- one candidate is selected;
- its score is read only on the held-out fold.

The mean held-out result is the nested calibration estimate. The separate
full-OOF winner is used only to define a final operating point after the nested
estimate has been recorded.

## Required experiment ledger

For every serious variant retain:

- git commit SHA;
- config file;
- fold checkpoints + SHA-256;
- fold PQ/SQ/RQ and TP/FP/FN;
- macro and micro OOF PQ;
- matched IoU/Dice quantiles;
- one-to-many / many-to-one counts;
- runtime and GPU;
- whether it was kept or rejected;
- one-sentence reason.

Negative results belong in the final report. They prove that model choices were
measured rather than guessed.

## Leaderboard discipline

The leaderboard is not a hyperparameter optimizer. A leaderboard submission
should confirm a frozen OOF decision, not create one.
