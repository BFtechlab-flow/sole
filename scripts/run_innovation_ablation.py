"""Progressive V2 innovation ablation (baseline_v2 + ten cumulative additions).

Each step is compared with the previous one (paired bootstrap of the official
micro PQ). Use ``--folds 0 --epochs N`` for a cheap screening pass first.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `src` imports work

from scripts._ablation_cli import main

INNOVATIONS = [
    ("baseline_v2", {
        "model.advanced.enabled": True,
        "model.advanced.limb_gate": False,
        "model.advanced.multiscale_context": False,
        "model.advanced.frequency_branch": False,
        "model.advanced.width_head": False,
        "model.advanced.curvature_head": False,
        "model.advanced.endpoint_head": False,
        "model.advanced.junction_head": False,
        "model.advanced.uncertainty_head": False,
        "loss.width": 0.0,
        "loss.curvature": 0.0,
        "loss.endpoint": 0.0,
        "loss.junction": 0.0,
        "loss.centerline_ce": 0.0,
        "loss.radius_balance": 0.0,
        "loss.uncertainty_region": 0.0,
        "inference.bridge_width_weight": 0.0,
    }),
    ("01_limb_gate", {"model.advanced.limb_gate": True}),
    ("02_multiscale_context", {"model.advanced.multiscale_context": True}),
    ("03_frequency_branch", {"model.advanced.frequency_branch": True}),
    ("04_physical_width", {
        "model.advanced.width_head": True,
        "loss.width": 0.05,
        "inference.bridge_width_weight": 0.35,
    }),
    ("05_curvature", {
        "model.advanced.curvature_head": True,
        "loss.curvature": 0.025,
    }),
    ("06_endpoint", {
        "model.advanced.endpoint_head": True,
        "loss.endpoint": 0.035,
    }),
    ("07_junction", {
        "model.advanced.junction_head": True,
        "loss.junction": 0.020,
    }),
    ("08_centerline_ce", {"loss.centerline_ce": 0.040}),
    ("09_radius_balance", {"loss.radius_balance": 0.040}),
    ("10_uncertainty", {
        "model.advanced.uncertainty_head": True,
        "loss.uncertainty_region": 0.030,
    }),
]



if __name__ == "__main__":
    main(INNOVATIONS, "configs/filanet.yaml", "artifacts/innovation_ablations", cumulative=True, description=__doc__)
