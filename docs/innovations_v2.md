# FILA-Net V2 — 10 ablation-first innovations

These additions are deliberately modular. None is considered a competition improvement until leakage-safe OOF PQ demonstrates it.

1. **Solar limb-aware gating** — learns a radial gain from the existing disk-radius input so the decoder can adapt to center-to-limb appearance changes.
2. **Multi-scale dilated context** — depthwise dilated branches increase the receptive field for long curved filaments without destroying native-resolution detail.
3. **Frequency residual branch** — injects explicit high-frequency and ridge evidence for thin dark threads.
4. **Physical-width prediction** — predicts a segmentation-derived diameter proxy from raw EDT, allowing width-aware endpoint reconnection.
5. **Curvature prediction** — predicts local tangent variation derived from the skeleton orientation field.
6. **Endpoint prediction** — supervises degree-1 skeleton keypoints to improve broken-filament reasoning.
7. **Junction prediction** — supervises high-degree skeleton points to reduce incorrect merges/splits around complex topology.
8. **Centerline-aware cross entropy** — increases learning pressure around the derived skeleton; inspired by centerline-bound topology objectives rather than claiming an exact reproduction of a published loss.
9. **Radius-balanced topology loss** — reweights topology errors so thin structures are not overwhelmed by thick ones; inspired by radius-aware tubular segmentation losses.
10. **Aleatoric uncertainty learning** — predicts ambiguity and uses it to attenuate noisy region supervision while penalizing unbounded uncertainty.

Additional implementation glue:
- adaptive feature mixer for base/context/frequency evidence;
- width-compatible graph bridging;
- gradient clipping + cosine LR schedule;
- progressive 11-stage ablation (`baseline_v2` + ten single additions).

## Rule-safe training strategy

- Public GONG H-alpha imagery may be used only as **unlabeled** domain data for self-supervised warm-up.
- Supervised filament learning must use the competition-provided segmentation annotations.
- No external MAGFiLO ground-truth masks are consumed.

## Commands

Real GONG preprocessing:

```bash
python scripts/real_sun_demo.py --output-dir artifacts/real_sun
```

Unlabeled GONG masked-reconstruction warm-up:

```bash
python scripts/pretrain_gong_ssl.py --output artifacts/pretrain/gong_ssl_encoder.pt
```

Transfer the 4-channel encoder into supervised training:

```bash
python scripts/train.py --fold 0 --encoder-weights artifacts/pretrain/gong_ssl_encoder.pt
```

Generate the innovation experiment matrix:

```bash
python scripts/run_innovation_ablation.py
```

Run it on a GPU machine that has the official MAGFiLO competition data:

```bash
python scripts/run_innovation_ablation.py --execute
```
