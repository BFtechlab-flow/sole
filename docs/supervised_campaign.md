# Supervised competition campaign

This campaign separates three questions that must not be conflated:

1. Does public unlabeled GONG H-alpha domain adaptation improve FILA-Net over the same ImageNet initialization?
2. Which of the 10 V2 innovations improve leakage-safe OOF PQ?
3. Which post-processing parameters transfer across held-out folds?

## A. Build the solar-domain encoder

Use an ImageNet-initialized 4-channel encoder, then adapt it with masked reconstruction on public NOAA/SWPC GONG H-alpha images:

```bash
python scripts/pretrain_gong_ssl.py \
  --imagenet-pretrained \
  --output artifacts/pretrain/gong_ssl_imagenet.pt \
  --manifest artifacts/pretrain/gong_ssl_imagenet.json \
  --max-images 64 \
  --min-images 48 \
  --epochs 10 \
  --steps-per-epoch 128 \
  --image-size 384
```

The exact corpus size/epochs should be chosen for the available GPU budget. Preserve the JSON manifest and source URLs. No external MAGFiLO ground-truth labels are used here.

## B. Paired five-fold ImageNet vs GONG comparison

On a GPU environment that has the official MAGFiLO competition data:

```bash
python scripts/run_supervised_campaign.py \
  --config configs/filanet.yaml \
  --encoder-weights artifacts/pretrain/gong_ssl_imagenet.pt \
  --output-dir artifacts/supervised_campaign \
  --execute
```

This trains the same FILA-Net V2 architecture twice across all five grouped folds:

- `v2_imagenet`: configured timm ImageNet initialization;
- `v2_imagenet_gong`: the same encoder after GONG self-supervised adaptation.

Architecture, seed, folds and post-processing are held fixed. The campaign creates OOF caches and per-record PQ results for both experiments, then runs a paired bootstrap comparison.

Primary decision signal: mean paired OOF PQ delta and its 95% bootstrap interval. Do not infer a competition gain from SSL reconstruction loss alone.

## C. Innovation ablation

Only after choosing the initialization, run the 10 innovations with the same starting encoder for every variant:

```bash
python scripts/run_innovation_ablation.py \
  --config configs/filanet.yaml \
  --encoder-weights artifacts/pretrain/gong_ssl_imagenet.pt \
  --output-dir artifacts/supervised_campaign/innovation_ablations_gong \
  --execute
```

Keep a feature only if leakage-safe OOF evidence supports it. Then perform nested post-processing calibration and final five-fold ensemble inference.
