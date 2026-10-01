# FILA-Net V3 cross-domain research backlog

This document records research directions only. None of the items below is kept in the final competition model unless leakage-safe OOF ablation improves the official-protocol PQ and/or qualitative error profile.

## Three ideas promoted into V3 implementation

1. **H0 persistent-homology topology loss** — from topological medical-image segmentation. FILA-Net uses a downsampled superlevel filtration to identify persistent connected components and train on the corresponding critical pixels. This targets fragmentation directly. It is limited to H0 (connected components), not full H1 topology.
2. **Geodesic graph repair** — from vessel/road tracing and path planning. Candidate endpoint pairs are linked by a minimum-cost path through the region/boundary probability field rather than a forced straight segment.
3. **Discriminative instance embedding** — from proposal-free instance segmentation. Pixels from the same filament are pulled together in embedding space and different filament centers are pushed apart. The embedding also becomes evidence for endpoint bridging.

## Ten additional cross-domain ideas to ablate later

### 1. Diffusion-MRI tractography: recurrent streamline propagation
Instead of reconstructing a whole centerline only from a thresholded heatmap, seed streamlines at confident filament points and propagate them step by step using the local orientation field plus image features and previous direction. Tractography literature treats fibers as continuous trajectories and explicitly models local/global direction consistency.

Candidate experiment: a lightweight GRU/Transformer propagator trained on skeleton trajectories generated from MAGFiLO masks. Compare fragmentation and overreach against watershed-only reconstruction.

### 2. Autonomous-driving lane detection: Bézier filament representation
Lane detectors model long continuous lanes as parametric Bézier curves rather than dense masks. A filament can similarly be approximated by one or several low-order Bézier segments. This gives a strong smoothness/continuity prior and a compact representation for long structures.

Candidate experiment: auxiliary head predicts 4 control points for long simple filaments; use only when curve-fit confidence is high.

### 3. Lane-graph extraction: direct vertex/edge prediction
Road and lane work increasingly predicts graph vertices and edge probabilities directly. FILA-Net already predicts endpoints and junctions, so a small graph transformer could consume those points plus local image embeddings and predict which vertices should be connected.

Candidate experiment: replace hand-designed endpoint thresholds with learned edge existence probabilities while preserving cycle constraints.

### 4. Fingerprint recognition: orientation-coherence confidence
Fingerprint systems rely heavily on ridge orientation fields and reject/repair regions where orientation coherence is poor. Solar filaments are also ridge-like. Add a local orientation-coherence target/feature so graph repair trusts smooth directional fields and downweights chaotic texture.

Candidate experiment: structure-tensor coherence channel plus coherence-weighted orientation loss.

### 5. Seismic fault interpretation: multi-scale coherence / discontinuity attributes
Seismic faults are thin extended structures embedded in highly textured data. Geophysical workflows use coherence/semblance-like attributes to expose structures that raw intensity hides.

Candidate experiment: add fixed multi-scale structure-tensor/coherence channels beside CLAHE/ridge, then ablate their effect on thin-filament recall.

### 6. Active contours / level sets: learned boundary evolution
After coarse instance reconstruction, evolve each boundary using image gradients plus learned region/boundary energies. This can sharpen filament widths without changing the centerline identity.

Candidate experiment: 3–5 differentiable contour-refinement iterations only around predicted boundaries; measure matched IoU/Dice and merge errors.

### 7. High-resolution instance segmentation: boundary-patch refinement
Modern instance segmentation often reprocesses only uncertain boundary patches at higher resolution. FILA-Net could identify low-confidence edge bands and run a small refinement network there instead of increasing full-image resolution.

Candidate experiment: 256-pixel boundary crops with a tiny refinement head; target only matched-instance IoU improvements.

### 8. Sequential map drawing: autoregressive graph growth
Recent lane-topology methods build a graph one node/edge at a time. A filament graph could be grown from the most confident endpoint, predicting the next point and stop/branch action until the filament is complete.

Candidate experiment: serialize GT skeleton graphs by depth-first traversal and train a small decoder; use only as a reconstruction challenger, not as the first production path.

### 9. Mixture-of-experts: thin-vs-thick filament specialists
A single decoder must handle tiny threads and broad dark structures. Use two lightweight experts conditioned on predicted width/context, with a learned gate. The thin expert optimizes centerline continuity; the thick expert optimizes boundaries/area.

Candidate experiment: shared encoder/FPN, two small decoder blocks, width-conditioned gate. Compare by filament-width bins.

### 10. Tracking/data association: minimum-cost global matching of fragments
Multi-object tracking avoids greedy local links by solving global association. Endpoint fragments can be treated the same way: build a compatibility matrix from distance, orientation, width, region evidence and embeddings, then solve a global minimum-cost matching instead of accepting candidates greedily.

Candidate experiment: Hungarian/min-cost-flow association on endpoint graph, constrained to avoid same-component cycles; compare fragmentation and many-to-one diagnostics.

## Research anchors

- Clough et al., persistent-homology topological loss: https://arxiv.org/abs/1910.01877
- De Brabandere et al., discriminative instance embeddings: https://arxiv.org/abs/1708.02551
- SAM-Road graph extraction: https://openaccess.thecvf.com/content/CVPR2024W/SG2RL/html/Hetang_Segment_Anything_Model_for_Road_Network_Graph_Extraction_CVPRW_2024_paper.html
- BézierLaneNet: https://openaccess.thecvf.com/content/CVPR2022/html/Feng_Rethinking_Efficient_Lane_Detection_via_Curve_Modeling_CVPR_2022_paper.html
- BGFormer / Bézier graphs: https://openaccess.thecvf.com/content/CVPR2024/html/Blayney_Bezier_Everywhere_All_at_Once_Learning_Drivable_Lanes_as_Bezier_CVPR_2024_paper.html
- SeqGrowGraph: https://openaccess.thecvf.com/content/ICCV2025/html/Xie_SeqGrowGraph_Learning_Lane_Topology_as_a_Chain_of_Graph_Expansions_ICCV_2025_paper.html
- Boundary Patch Refinement: https://openaccess.thecvf.com/content/CVPR2021/html/Tang_Look_Closer_To_Segment_Better_Boundary_Patch_Refinement_for_Instance_CVPR_2021_paper.html
- Deep Structured Active Contours: https://openaccess.thecvf.com/content_cvpr_2018/html/Marcos_Learning_Deep_Structured_CVPR_2018_paper.html
- dMRI tractography implementation considerations: https://pubmed.ncbi.nlm.nih.gov/37066284/
- Bundle-constrained streamline tractography: https://pubmed.ncbi.nlm.nih.gov/26342757/
