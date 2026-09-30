# SOLAR — FILA-Net

Competition-oriented PyTorch pipeline for the **Solar Filament Segmentation Challenge 2026**.

## What this repository contains

- grouped cross-validation by physical observation
- multi-head FILA-Net: region, derived centerline, boundary, distance transform
- segmentation-only supervision: no spine/chirality/bbox metadata used as GT
- clDice-style topology loss
- full-resolution overlapping tiled inference
- conservative centerline fragment reconnection
- marker-controlled watershed instance reconstruction
- local Panoptic Quality diagnostics
- TTA and multi-fold ensemble inference
- COCO compressed RLE with round-trip verification

## Expected dataset layout

```text
MAGFiLO_1.0_Kaggle_2026/
├── train/
│   ├── train_images/
│   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
└── test/
    └── test_images/
```

Edit `data.root` in `configs/filanet.yaml`.

## Install

```bash
pip install -r requirements.txt
```

## Audit data

```bash
python scripts/audit_data.py --config configs/filanet.yaml
```

## Train

```bash
python scripts/train.py --config configs/filanet.yaml --fold 0
```

Repeat for folds 0–4.

## Validate

```bash
python scripts/validate.py --config configs/filanet.yaml --fold 0 --weights checkpoints/fold0_best.pt
```

## Submission

```bash
python scripts/infer.py --config configs/filanet.yaml \
  --weights checkpoints/fold0_best.pt checkpoints/fold1_best.pt checkpoints/fold2_best.pt checkpoints/fold3_best.pt checkpoints/fold4_best.pt \
  --output submission.csv
```

## Important

Thresholds in the YAML are starting values, not claimed optimal values. Tune them only on leakage-safe OOF predictions. Before a final Kaggle submission, compare `src/metric.py` against the organizers' latest evaluation implementation and align any competition-specific edge cases.
