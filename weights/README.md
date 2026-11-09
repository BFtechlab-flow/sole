# Trained weights

These are the weights of the final submission (public leaderboard 0.39,
submission sha256 `3c851c45091e0268d59d8f8a896a57eafea9fe88ff66137db4e6422f2d0d4291`).

| file | what | where |
|---|---|---|
| `fold{0..4}_best.pt` | ConvNeXt-T + FPN, one model per fold (122 MB each) | GitHub Release `v1.0` (as `.zip`) |
| `convnext_small_fold{0..4}_best.pt` | ConvNeXt-S + FPN, one model per fold (209 MB each) | GitHub Release `v1.0` (as `.zip`) |
| `rescorer.json` | learned filament filter (300 gradient-boosted trees, 0.4 MB) | this folder |
| `manifest.json` | URL, SHA-256, size, best epoch and validation PQ of each checkpoint | this folder |

In the release the checkpoints are named `.zip` (a PyTorch checkpoint is a
zip archive and the upload kept that extension); the bytes are the `.pt`
files, and `tools/weights.py fetch` saves them as `.pt` after checking the
SHA-256 of `manifest.json`.

Each checkpoint is the epoch with the best official micro PQ on its held-out
fold (`validation_micro_pq` in the manifest) and stores its own architecture,
so the ten files form one ensemble without extra settings. The reconstruction
parameters and the filter threshold are in `configs/filanet_tuned.yaml`; the
fold assignment is `configs/folds_day5.csv`.

Why the filter is JSON: it was trained on Kaggle with scikit-learn 1.6.1, and
a scikit-learn pickle loads only with the version that wrote it.
`tools/export_rescorer.py` converts it to plain trees that `src/rescore.py`
evaluates with numpy; the predictions are identical (maximum difference 0 on
20,000 random rows, see the tool and `tests/test_rescore.py`). The original
pickle had sha256 `8f794d1bba90bb71337fd7017f7273740ed338fb4e6f95cdc12f49b6d4f741de`.

## Reproducing the submission

```bash
pip install -r requirements.txt
python tools/weights.py fetch --dest checkpoints     # download + SHA-256 check (1.65 GB)
# set data.root in configs/filanet_tuned.yaml to your copy of the competition data
python scripts/infer.py --config configs/filanet_tuned.yaml \
  --weights checkpoints/fold{0,1,2,3,4}_best.pt checkpoints/convnext_small_fold{0,1,2,3,4}_best.pt \
  --output submission.csv
```

The submitted file was produced on Kaggle (2 × T4 for the network maps, CPU
for the post-processing) with Python 3.13.15, torch 2.11.0, timm 1.0.29,
numpy 2.1.3, OpenCV 4.14.0, SciPy 1.16.3, scikit-image 0.25.2. The
network runs with mixed precision, so another GPU or library version can move
a few boundary pixels.

## Publishing the checkpoints (done once)

On GitHub: *Releases → Draft a new release*, tag `v1.0`, attach the ten `.pt`
files from the notebook output (`sole/artifacts/release/`), publish. Then
`python tools/weights.py verify --dest <folder with the files>` checks them
against `manifest.json`.
