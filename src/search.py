from __future__ import annotations

import numpy as np


def nested_candidate_selection(score_matrix):
    """Leakage-safe outer-fold selection.

    ``score_matrix`` is shape [candidate, fold]. For each held-out fold, choose
    the candidate with the best mean score on all *other* folds, then report its
    held-out score. This quantifies post-processing selection without evaluating
    a candidate on the fold that selected it.
    """
    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] < 2:
        raise ValueError("score_matrix must be [candidate, fold] with >=2 folds")
    selections = []
    for heldout in range(scores.shape[1]):
        train_folds = [i for i in range(scores.shape[1]) if i != heldout]
        train_mean = scores[:, train_folds].mean(axis=1)
        candidate = int(np.argmax(train_mean))
        selections.append(
            {
                "heldout_fold": int(heldout),
                "candidate": candidate,
                "selection_score": float(train_mean[candidate]),
                "heldout_score": float(scores[candidate, heldout]),
            }
        )
    nested = float(np.mean([x["heldout_score"] for x in selections]))
    final_candidate = int(np.argmax(scores.mean(axis=1)))
    return {
        "nested_macro_pq": nested,
        "outer_selections": selections,
        "final_candidate": final_candidate,
        "final_oof_score": float(scores[final_candidate].mean()),
    }
