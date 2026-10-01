# Evaluation protocol

## Organizer references

- Competition evaluation/announcements:
  https://www.kaggle.com/competitions/filament-segmentation-2026/overview/announcements
- Self Evaluation notebook:
  https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
- Panoptic Segmentation / PQ paper:
  https://arxiv.org/abs/1801.00868

## Semantics implemented here

The published competition definition is:

`PQ = sum(IoU of matched pairs) / (TP + 0.5*FP + 0.5*FN)`

A pair is a true positive only when **IoU > 0.50**. Equality at 0.50 is not a
match.

MAGFiLO can contain several independent annotation records for one physical
JPEG. FILA-Net therefore:

1. predicts a physical JPEG once;
2. reconstructs one prediction set;
3. scores that same prediction set independently against each annotation record;
4. reports the mean record PQ (`macro_pq`) as the primary local selection metric;
5. additionally reports `micro_pq` from globally accumulated TP/FP/FN.

No annotator-union target is used for the official-protocol metric.

## Matching

For valid non-overlapping panoptic instances, IoU > 0.5 guarantees uniqueness.
`src/pq_official.py` therefore selects all pairs above the strict threshold and
raises if a non-unique match is observed. This deliberately avoids silently
introducing a Hungarian assignment rule that is not part of the published
formula.

## Additional diagnostics

The final judging rubric also cares about segmentation distributions and
fragmentation/merging. We therefore retain:

- matched IoU distribution;
- matched Dice distribution (`Dice = 2*IoU/(1+IoU)` for a matched pair);
- one-to-many relations (one GT overlaps several predictions);
- many-to-one relations (one prediction overlaps several GT instances);
- TP, FP and FN;
- SQ and RQ;
- record-level bootstrap uncertainty.

The relation threshold is diagnostic and lives separately in
`metric.relation_iou_threshold`; it is **not** part of PQ.

## Reference parity

The organizer's Self Evaluation notebook is the final authority. If a newer
version changes an edge case, update `src/pq_official.py`, add a regression test
that reproduces the notebook case, and re-run all OOF calibration. Do not tune
against leaderboard deltas to infer metric semantics.
