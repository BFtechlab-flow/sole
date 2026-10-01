# SOLAR — FILA-Net

Competition-oriented PyTorch pipeline for the **Solar Filament Segmentation Challenge 2026**.

FILA-Net is an instance-reconstruction system for 2048×2048 GONG H-alpha images:

`image → multi-head filament fields → topology/orientation reasoning → graph repair → watershed instances → COCO RLE`

## Current capabilities

- leakage-safe grouped 5-fold CV by physical GONG observation
- independent scoring of every MAGFiLO annotator record
- ConvNeXt/FPN multi-head predictions: region, centerline, boundary, distance, orientation
- segmentation-derived auxiliary supervision only
- clDice-style topology loss
- axial orientation field `[cos(2θ), sin(2θ)]`
- overlapping native-resolution tiled inference
- flip TTA with correct orientation transforms
- orientation-aware conservative endpoint graph bridging
- marker-controlled watershed reconstruction
- competition-aligned PQ diagnostics with strict `IoU > 0.50`
- matched IoU/Dice distributions + one-to-many / many-to-one diagnostics
- OOF prediction cache
- nested post-processing threshold calibration
- single-variable ablation orchestration
- five-fold ensemble inference
- hard submission/RLE validation + SHA-256 verification
- end-to-end Jupyter notebook
- exact pinned Python package versions

## Important metric note

The organizer's public evaluation definition uses Panoptic Quality:

`PQ = Σ IoU(matched pairs) / (TP + 0.5 FP + 0.5 FN)`

with a **strict IoU > 0.50** instance match. MAGFiLO has multiple independent
annotation records for some physical images, so one prediction set is evaluated
against each annotation record separately. `src/pq_official.py` implements these
published semantics and exposes both macro and micro summaries.

The organizer's **Self Evaluation notebook is the final authority** if its
implementation changes:
https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

## Expected dataset layout

```text
MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
└── test/
    └── test_images/
```

Set `data.root` in `configs/filanet.yaml`.

## Environment

Tested CI environment:

- Python 3.11.16
- exact package versions in `requirements.txt`

```bash
python -m pip install -r requirements.txt
```

## 1. Audit data and folds

```bash
python scripts/audit_data.py \
  --config configs/filanet.yaml \
  --strict \
  --output artifacts/data_audit.json
```

For a slower complete mask-decode audit add `--decode-all`.

## 2. Train five folds

```bash
for FOLD in 0 1 2 3 4; do
  python scripts/train.py \
    --config configs/filanet.yaml \
    --fold "$FOLD" \
    --output checkpoints
done
```

Best checkpoints are selected by macro PQ over the fold's independent
annotation records, not by training loss.

## 3. Cache OOF predictions

```bash
python scripts/cache_oof.py \
  --config configs/filanet.yaml \
  --weights-pattern 'checkpoints/fold{fold}_best.pt' \
  --cache-dir artifacts/oof_cache
```

Each physical image is inferred only once per fold even when several annotators
labeled it.

## 4. Full OOF evaluation

```bash
python scripts/evaluate_oof.py \
  --config configs/filanet.yaml \
  --cache-dir artifacts/oof_cache \
  --output artifacts/oof_metrics.json \
  --records-csv artifacts/oof_records.csv
```

Outputs include macro/micro PQ, SQ/RQ, TP/FP/FN, matched IoU/Dice
distributions, fragmentation/merge diagnostics and bootstrap uncertainty.

## 5. Leakage-safe post-processing calibration

```bash
python scripts/optimize_postprocess.py \
  --config configs/filanet.yaml \
  --search-config configs/postprocess_search.yaml \
  --cache-dir artifacts/oof_cache \
  --output artifacts/postprocess_search.json \
  --write-config configs/filanet_tuned.yaml
```

For every outer fold, the parameter candidate is selected using the **other**
folds only. The nested score is the number to use when judging whether tuning
actually transfers.

## 6. Ablations

Generate the campaign without spending GPU time:

```bash
python scripts/run_ablation.py
```

Run it on a GPU machine with the official data:

```bash
python scripts/run_ablation.py --execute
```

The sequence changes one capability at a time:

`baseline → +topology → +orientation → +graph bridge → +TTA`

Do not keep a feature merely because it sounds sophisticated. Keep it only if
paired OOF evidence supports it.

## 7. Final ensemble submission

```bash
python scripts/infer.py \
  --config configs/filanet_tuned.yaml \
  --weights \
    checkpoints/fold0_best.pt \
    checkpoints/fold1_best.pt \
    checkpoints/fold2_best.pt \
    checkpoints/fold3_best.pt \
    checkpoints/fold4_best.pt \
  --output submission.csv
```

`infer.py` automatically validates the CSV and writes
`submission.verification.json`.

A standalone gate is also available:

```bash
python scripts/validate_submission.py submission.csv \
  --test-images /path/to/MAGFiLO_1.0_Kaggle_2026/test/test_images
```

## End-to-end notebook

Open:

```text
notebooks/filanet_end_to_end.ipynb
```

It walks through audit → five-fold training → OOF → calibration → ablation →
ensemble inference → submission verification.

## Competition discipline

- Never tune from the public/private leaderboard.
- Never merge annotators into one validation target for PQ reporting.
- Never report a feature as an improvement until it survives leakage-safe OOF.
- Preserve all ablation failures; they are useful evidence for the technical report.
- Do not use test-set labels or external test-overlapping annotations.
- Treat `configs/filanet_tuned.yaml` as valid only after it has been generated from OOF caches.

See `docs/experiment_protocol.md` and `docs/final_submission_checklist.md`.
