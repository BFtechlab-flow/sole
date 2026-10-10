"""V3 ablation: V2 reference, H0 topology loss only, instance embedding only,
geodesic repair only, all V3 components. Same folds, seed, encoder and
post-processing for every variant; ``geodesic_only`` changes inference only
and reuses the V2-reference checkpoints and OOF cache.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `src` imports work

from scripts._ablation_cli import main

V3_VARIANTS = [
    (
        "v2_reference",
        {
            "model.advanced.instance_embedding_head": False,
            "loss.h0_persistence": 0.0,
            "loss.instance_embedding": 0.0,
            "inference.bridge_geodesic": False,
            "inference.bridge_embedding_weight": 0.0,
            "inference.bridge_min_embedding_similarity": -1.0,
        },
    ),
    (
        "h0_only",
        {
            "model.advanced.instance_embedding_head": False,
            "loss.h0_persistence": 0.015,
            "loss.instance_embedding": 0.0,
            "inference.bridge_geodesic": False,
            "inference.bridge_embedding_weight": 0.0,
            "inference.bridge_min_embedding_similarity": -1.0,
        },
    ),
    (
        "embedding_only",
        {
            "model.advanced.instance_embedding_head": True,
            "loss.h0_persistence": 0.0,
            "loss.instance_embedding": 0.035,
            "inference.bridge_geodesic": False,
            "inference.bridge_embedding_weight": 0.35,
            "inference.bridge_min_embedding_similarity": 0.15,
        },
    ),
    (
        "geodesic_only",
        {
            "model.advanced.instance_embedding_head": False,
            "loss.h0_persistence": 0.0,
            "loss.instance_embedding": 0.0,
            "inference.bridge_geodesic": True,
            "inference.bridge_embedding_weight": 0.0,
            "inference.bridge_min_embedding_similarity": -1.0,
        },
    ),
    (
        "v3_all",
        {
            "model.advanced.instance_embedding_head": True,
            "loss.h0_persistence": 0.015,
            "loss.instance_embedding": 0.035,
            "inference.bridge_geodesic": True,
            "inference.bridge_embedding_weight": 0.35,
            "inference.bridge_min_embedding_similarity": 0.15,
        },
    ),
]



if __name__ == "__main__":
    main(V3_VARIANTS, "configs/filanet_v3.yaml", "artifacts/v3_ablations", cumulative=False, description=__doc__)
