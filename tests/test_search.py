import numpy as np

from src.search import nested_candidate_selection


def test_nested_selection_never_uses_heldout_fold_for_selection():
    scores = np.array([
        [0.30, 0.30, 0.30, 0.30, 0.30],
        [0.90, 0.10, 0.10, 0.10, 0.10],
    ], dtype=float)
    result = nested_candidate_selection(scores)
    fold0 = result["outer_selections"][0]
    assert fold0["candidate"] == 0
    assert fold0["heldout_score"] == 0.30
    assert result["final_candidate"] == 0
