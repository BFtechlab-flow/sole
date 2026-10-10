"""Core progressive ablation: baseline -> +topology -> +orientation -> +graph
bridge -> +TTA (-> +full-resolution refiner, experimental).

Inference-only steps (bridge, TTA) reuse the previous checkpoints, so they
cost no training. Every step is compared with the previous one by a paired
bootstrap of the official micro PQ.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `src` imports work

from scripts._ablation_cli import main

ABLATIONS = [
    ("baseline", {
        "loss.topology": 0.0,
        "loss.orientation": 0.0,
        "inference.bridge_max_distance": 0,
        "inference.tta": False,
    }),
    ("plus_topology", {"loss.topology": 0.05}),
    ("plus_orientation", {"loss.orientation": 0.08}),
    ("plus_bridge", {"inference.bridge_max_distance": 28}),
    ("plus_tta", {"inference.tta": True}),
    ("plus_fullres", {"model.full_res_refine": True}),
]

if __name__ == "__main__":
    main(ABLATIONS, "configs/filanet.yaml", "artifacts/ablations", cumulative=True, description=__doc__)
