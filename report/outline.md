# Technical report working outline

Use the organizer's required Overleaf template for the final four-page PDF.
This file is a content checklist, not a replacement template.

## 1. Problem and contribution

- Solar filament instance segmentation in 2048×2048 GONG H-alpha imagery.
- PQ-aware formulation.
- Key idea: dense region + topology + orientation fields, followed by
  conservative graph repair and instance watershed reconstruction.
- State the exact final OOF PQ only after the five-fold run is complete.

## 2. Data and validation

- MAGFiLO competition split.
- Multiple independent annotator records for some physical JPEGs.
- Grouped split prevents same-observation leakage.
- Explain macro record PQ and report micro PQ as secondary.
- Describe nested post-processing calibration.

## 3. Method

- Four-channel input: robust-normalized intensity, CLAHE, ridge response, radial coordinate.
- ConvNeXt Tiny feature pyramid.
- Region / centerline / boundary / distance / axial orientation heads.
- BCE + Dice + distance + clDice + orientation losses.
- Native-resolution tiled inference and flip TTA.
- Endpoint graph bridging.
- Marker-controlled watershed.
- COCO compressed RLE.

## 4. Experiments

Required table:

| Variant | Macro OOF PQ | Δ | SQ | RQ | TP | FP | FN | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | TBD | — | TBD | TBD | TBD | TBD | TBD | reference |
| + topology | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| + orientation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| + graph bridge | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| + TTA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Also include:
- per-fold PQ variation;
- nested calibration estimate;
- matched IoU/Dice quantiles;
- one-to-many / many-to-one counts;
- runtime.

## 5. Qualitative morphology

Include representative:
- long filament recovered correctly;
- faint/small filament;
- fragmentation corrected by graph bridge;
- failure from false positive;
- failure from over-merge;
- annotator disagreement example.

## 6. Reproducibility and limitations

- exact environment;
- seed and config;
- checkpoint hashes;
- full notebook;
- dataset is not redistributed;
- no test labels/external test-overlap annotations;
- discuss multi-annotator ceiling and remaining recognition errors.
