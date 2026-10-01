import numpy as np

from src.campaign import experiment_commands, paired_bootstrap


def test_paired_bootstrap_detects_uniform_gain():
    baseline = np.array([0.1, 0.2, 0.4, 0.7], dtype=np.float64)
    candidate = baseline + 0.05
    report = paired_bootstrap(baseline, candidate, iterations=200, seed=7)
    assert abs(report["mean_delta"] - 0.05) < 1e-12
    assert abs(report["bootstrap_ci95"][0] - 0.05) < 1e-12
    assert abs(report["bootstrap_ci95"][1] - 0.05) < 1e-12
    assert report["bootstrap_probability_positive"] == 1.0


def test_paired_bootstrap_rejects_unpaired_shapes():
    try:
        paired_bootstrap([0.1, 0.2], [0.1], iterations=10)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_experiment_commands_change_only_encoder_initialization():
    base = experiment_commands("cfg.yaml", "out", "base", 2, encoder_weights=None, epochs=3)
    ssl = experiment_commands("cfg.yaml", "out", "ssl", 2, encoder_weights="gong.pt", epochs=3)
    assert len(base) == len(ssl) == 4
    for i in range(2):
        assert "--encoder-weights" not in base[i]
        assert ssl[i][-2:] == ["--encoder-weights", "gong.pt"]
        assert base[i][base[i].index("--epochs") + 1] == "3"
        assert ssl[i][ssl[i].index("--epochs") + 1] == "3"
    assert base[-2][1] == ssl[-2][1] == "scripts/cache_oof.py"
    assert base[-1][1] == ssl[-1][1] == "scripts/evaluate_oof.py"
