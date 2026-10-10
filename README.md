# FILA-Net — solar filament instance segmentation

Solution code for the **Solar Filament Segmentation Challenge 2026**
([Kaggle](https://www.kaggle.com/competitions/filament-segmentation-2026),
IEEE BigData Cup 2026): every solar filament in a 2048 × 2048 GONG H-alpha
image is segmented as a separate instance and written as a COCO RLE.

```text
H-alpha JPEG ─► 4 input channels ─► ConvNeXt-T + FPN (multi-head) ─► dense fields
(normalised, CLAHE, ridge, radial)   region · centerline · boundary · distance
                                     orientation · width · endpoint/junction · …
dense fields ─► thresholds ─► centerline seeds ─► endpoint graph repair ─► marker watershed
            ─► hole filling / dilation / confidence filter ─► disjoint instances ─► RLE CSV
```

## Status — read this first

| item | state |
|---|---|
| competition metric (PQ) | implemented in `src/pq_official.py`; gives **identical** scores to the organizers' notebook functions on random multi-annotator scenes (`tools/verify_against_official_notebook.py`, `tests/test_official_parity.py`) |
| submission format | writer + strict validator (header, no quotes, unique ids, RLE decodes to 2048 × 2048, no empty/overlapping masks) |
| training / OOF / calibration / ablation / inference | implemented, tested end to end on a synthetic dataset (CPU, `tests/test_pipeline_smoke.py`, `tests/test_notebook_smoke.py`) |
| **supervised training on the official data** | **not run yet** — there is no trained checkpoint, no OOF score and no leaderboard score in this repository |

No number is reported here unless it was produced by the commands below on
the official data. Every score the pipeline writes is the official micro PQ.

## Competition rules this repository follows

Summary (details and sources in [`docs/COMPETITION_RULES.md`](docs/COMPETITION_RULES.md)):

* **Metric** — Panoptic Quality as in the organizers' *Self_Evaluation_Notebook*:
  each annotator's label set of an observation is a separate entry, a pair is a
  match when IoU **> 0.5**, TP/FP/FN and matched IoUs are summed over all
  entries and PQ is computed once (**micro**). Checkpoint selection, OOF
  evaluation, post-processing calibration and ablations all use this number.
* **Data** — only the annotations shipped with the competition are used for
  training. The public MAGFiLO release contains the test labels and is never
  downloaded. Unlabelled public GONG frames may be used for self-supervised
  pre-training only.
* **Inference** — each test image is predicted from that image alone (no
  neighbouring frame or other observation).
* **Deliverables (15 Nov 2026)** — 4-page report in the organizers' Overleaf
  template, this public repository with a fully pinned `requirements.txt`, and
  the notebook [`notebooks/filanet_end_to_end.ipynb`](notebooks/filanet_end_to_end.ipynb)
  that runs the entire pipeline. Trained weights are published as a GitHub
  Release and listed with their SHA-256 in [`weights/manifest.json`](weights/README.md).

## Quick start

```bash
python -m pip install -r requirements.txt     # Python 3.11, exact versions
python -m pytest -q                           # ~3 min on CPU, includes the end-to-end smoke tests
```

`tests/test_notebook_smoke.py` executes every cell of the notebook in smoke
mode; in Jupyter or Kaggle the notebook runs as is (`MODE = "smoke"` by
default, `"full"` for the official data).

The **smoke mode** runs the whole pipeline on a tiny synthetic dataset
(`tools/make_synthetic_dataset.py`: artificial curves, not solar data) to prove
that the code runs; its scores are meaningless.

The full run on the official data needs a GPU: the step-by-step plan for
Kaggle (sessions, time budget, resuming, what to submit) is in
[`docs/KAGGLE_RUNBOOK.md`](docs/KAGGLE_RUNBOOK.md).

## Pipeline commands

All scripts read `configs/filanet.yaml` (V2 reference model) unless
`--config` is given; `configs/filanet_v3.yaml` adds the V3 components. The
dataset root is found automatically under `/kaggle/input` or set with
`$FILANET_DATA_ROOT`.

```bash
# 1. audit the data + human baseline (annotator vs annotator, official PQ)
python scripts/audit_data.py --inter-annotator --output artifacts/data_audit.json

# 2. train the 5 folds (best epoch = best official PQ on the held-out fold; resumable)
for k in 0 1 2 3 4; do python scripts/train.py --fold $k --output checkpoints --max-hours 11; done

# 3. out-of-fold predictions (once per observation) and official evaluation
python scripts/cache_oof.py --weights-pattern 'checkpoints/fold{fold}_best.pt' --cache-dir artifacts/oof_cache
python scripts/evaluate_oof.py --cache-dir artifacts/oof_cache --export-dir artifacts/oof_export

# 4. nested post-processing calibration -> configs/filanet_tuned.yaml
python scripts/optimize_postprocess.py --cache-dir artifacts/oof_cache --write-config configs/filanet_tuned.yaml

# 5. ablations (dry run writes the plan; --execute runs it; screen with --folds 0 --epochs 8)
python scripts/run_ablation.py            # baseline → +topology → +orientation → +bridge → +TTA → +full-res
python scripts/run_innovation_ablation.py # the ten V2 components, one at a time
python scripts/run_v3_ablation.py         # H0 loss / instance embedding / geodesic repair

# 6. test inference with the fold ensemble -> submission.csv (+ checks and runtime report)
python scripts/infer.py --config configs/filanet_tuned.yaml \
  --weights checkpoints/fold{0,1,2,3,4}_best.pt --output submission.csv
python scripts/validate_submission.py submission.csv --test-images <data>/test/test_images

# 7. publish / fetch the trained weights (see weights/README.md)
python tools/weights.py manifest --checkpoints checkpoints/fold{0,1,2,3,4}_best.pt \
  --base-url https://github.com/BFtechlab-flow/sole/releases/download/v1.0/ --config configs/filanet_tuned.yaml
python tools/weights.py fetch --dest checkpoints
```

`scripts/compare_oof_experiments.py` compares two OOF runs with a paired
bootstrap of the micro PQ (whole observations resampled);
`scripts/pretrain_gong_ssl.py` adapts the encoder to unlabelled public GONG
frames (masked reconstruction) and `scripts/train.py --encoder-weights` loads it.

## Method in brief

* **Inputs** — robust-normalised intensity, CLAHE, a ridge (difference of
  Gaussians) response and the radial position on the disk.
* **Network** — ConvNeXt-Tiny (ImageNet) encoder, FPN decoder at stride 4,
  optional multi-scale dilated context, frequency branch and limb-aware gate;
  1×1 heads predict region, centerline, boundary, normalised distance, axial
  orientation `(cos 2θ, sin 2θ)` and, optionally, width, curvature, endpoints,
  junctions, aleatoric uncertainty and an instance embedding. All targets are
  derived from the segmentation polygons only.
* **Losses** — BCE + Dice (region, centerline), boundary BCE, distance
  regression, clDice topology loss, axial orientation loss, optional
  persistent-homology (H0) loss and discriminative embedding loss.
* **Validation** — 5 folds grouped by UT day; each epoch is scored with the
  official PQ on the held-out fold.
* **Inference** — 1024 px tiles with 256 px overlap blended by a Hann window,
  flip TTA (orientation transformed consistently), fold ensemble, float16
  autocast on GPU.
* **Instances** — thresholded region, skeletonised centerline seeds, endpoint
  graph repair (orientation / width / embedding compatibility, optional
  geodesic paths), marker-controlled watershed, then hole filling, dilation
  and a confidence filter. Under micro PQ an instance is worth keeping only if
  `P(match) × IoU > PQ / 2`, so the minimum confidence is calibrated on OOF
  predictions with a nested procedure.

Every optional component is switchable from the config and has an ablation
entry: a component stays in the final model only if the paired OOF comparison
supports it.

## Repository layout

```text
configs/     model, training, inference and search configurations (+ CPU smoke config)
docs/        competition rules, Kaggle runbook, protocols, Italian guide (GUIDA_IT.md)
notebooks/   filanet_end_to_end.ipynb — the whole pipeline (smoke / full modes)
report/      technical-report draft for the organizers' Overleaf template, pipeline figure
scripts/     command-line entry points (audit, train, OOF, calibration, ablations, inference)
src/         library code (data, model, losses, inference, reconstruction, metric, submission)
tests/       unit, parity and end-to-end tests (pytest)
tools/       synthetic dataset, metric parity check, weights manifest/fetch, figures
weights/     manifest of the published checkpoints (empty until training is done)
```

## License

Apache-2.0 (see `LICENSE` and `NOTICE`). Competition data: CC BY-NC 4.0, not
redistributed.
