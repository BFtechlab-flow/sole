"""Leakage-safe post-processing calibration on cached OOF predictions.

Two search modes, both scored with the official micro PQ:

* ``coordinate`` (default) - coordinate ascent over the parameters of
  ``space`` (in the listed order, ``rounds`` passes), starting from the
  current config. It is run once per outer fold on the OTHER folds (nested
  estimate) and once on all folds (final setting). A memo table shares every
  evaluated candidate between the searches, so the total cost is close to a
  single search.
* ``random`` - a seeded random sample of the grid plus the current setting,
  evaluated once; the nested estimate selects per outer fold among them.

The **nested estimate** (held-out folds scored with settings chosen without
them) is the honest number to report; the final setting is written into
``--write-config`` for test inference.

Long searches are resumable: ``--memo`` stores every evaluated candidate
(per-entry TP/FP/FN/IoU counts) after each evaluation, and ``--max-hours``
stops cleanly before the budget. Rerunning the same command replays the
already evaluated candidates instantly (the search is deterministic) and
continues where it stopped. The output JSON has ``"complete": false`` until
the whole search has run; the tuned config is only written when it is
complete.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `src` imports work

import argparse
import copy
import hashlib
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import yaml

from src.engine import load_records_and_folds, postprocess_params
from src.oof_eval import build_jobs, counts_array, default_workers, run_jobs
from src.utils import load_yaml


def _pq(s):
    den = s[..., 1] + 0.5 * s[..., 2] + 0.5 * s[..., 3]
    return np.where(den > 0, s[..., 0] / np.maximum(den, 1e-12), 0.0)


class BudgetExceeded(Exception):
    """Raised before evaluating new candidates once the time budget is used."""


class Evaluator:
    """Memoised evaluation of post-processing settings on all OOF entries.

    The memo (candidate -> per-entry counts) can be persisted with
    ``memo_path``; its ``signature`` ties it to the evaluated images, the metric
    settings and the OOF cache (weights of every fold), so a memo computed on
    other predictions is never reused.
    """

    def __init__(self, records, folds, cache_dir, base, metric_cfg, workers, max_images_per_fold, seed,
                 memo_path=None, deadline=None):
        self.base = dict(base)
        self.jobs = build_jobs(records, folds, cache_dir, [self.base], metric_cfg,
                               max_images_per_fold=max_images_per_fold, seed=seed)
        self.workers = workers
        self.memo = {}
        self.entry_folds = None
        self.passes = 0
        self.memo_path = Path(memo_path) if memo_path else None
        self.deadline = deadline
        self.signature = self._signature(cache_dir, metric_cfg)
        self.restored = 0
        if self.memo_path is not None and self.memo_path.exists():
            self._load()

    def _signature(self, cache_dir, metric_cfg):
        weights = {}
        for m in sorted(Path(cache_dir).glob("manifest*.json")):
            try:
                for fold, info in json.loads(m.read_text()).get("folds", {}).items():
                    weights[str(fold)] = info.get("weights_sha256")
            except (OSError, ValueError):
                continue
        self.oof_weights = weights
        payload = {"images": [(int(j[0]), j[1], len(j[2])) for j in self.jobs],
                   "metric": metric_cfg or {}, "oof_weights": weights}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def _load(self):
        with np.load(self.memo_path, allow_pickle=False) as data:
            if str(data["signature"]) != self.signature:
                print(f"memo {self.memo_path} was computed on other OOF predictions/images: ignored", flush=True)
                return
            tokens = [str(t) for t in data["tokens"]]
            counts = data["counts"]
            self.entry_folds = data["entry_folds"]
        self.memo = {t: counts[i] for i, t in enumerate(tokens)}
        self.restored = len(self.memo)
        print(f"restored {self.restored} evaluated candidate(s) from {self.memo_path}", flush=True)

    def _save(self):
        if self.memo_path is None or not self.memo:
            return
        self.memo_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.memo_path.with_name(self.memo_path.stem + ".tmp.npz")
        tokens = list(self.memo)
        np.savez_compressed(tmp, signature=np.array(self.signature), tokens=np.array(tokens),
                            counts=np.stack([self.memo[t] for t in tokens]), entry_folds=self.entry_folds,
                            oof_weights=np.array(json.dumps(self.oof_weights, sort_keys=True)))
        tmp.replace(self.memo_path)

    @staticmethod
    def token(params):
        return json.dumps(params, sort_keys=True, default=str)

    def evaluate(self, candidates):
        todo, seen = [], set()
        for c in candidates:
            t = self.token(c)
            if t not in self.memo and t not in seen:
                todo.append(c)
                seen.add(t)
        if todo:
            if self.deadline is not None and time.time() > self.deadline:
                raise BudgetExceeded
            jobs = [j[:4] + (todo,) + j[5:] for j in self.jobs]
            per_set = run_jobs(jobs, workers=self.workers)
            self.passes += 1
            for c, rows in zip(todo, per_set):
                f, _, scored, counts = counts_array(rows)
                if self.entry_folds is None:
                    self.entry_folds = f[scored]
                self.memo[self.token(c)] = counts[scored]
            self._save()
            print(f"  evaluated {len(todo)} new candidate(s); memo size {len(self.memo)}", flush=True)
        return [self.memo[self.token(c)] for c in candidates]

    def fold_sums(self, counts, folds_sel):
        mask = np.isin(self.entry_folds, list(folds_sel))
        return counts[mask].sum(axis=0)


def coordinate_search(ev, space, rounds, select_folds):
    current = dict(ev.base)
    cur_score = float(_pq(ev.fold_sums(ev.evaluate([current])[0], select_folds)))
    path = [{"step": "start", "pq": cur_score}]
    for r in range(int(rounds)):
        changed = False
        for key, values in space.items():
            cands = [dict(current, **{key: v}) for v in values]
            scores = [float(_pq(ev.fold_sums(c, select_folds))) for c in ev.evaluate(cands)]
            best = int(np.argmax(scores))
            if scores[best] > cur_score + 1e-9 and cands[best][key] != current.get(key):
                current, cur_score, changed = cands[best], scores[best], True
                path.append({"round": r, "key": key, "value": current[key], "pq": cur_score})
        if not changed:
            break
    return current, cur_score, path


def candidate_grid(search_space, max_candidates, seed, base):
    keys = list(search_space)
    values = [search_space[k] for k in keys]
    n_total = int(np.prod([len(v) for v in values])) if values else 1
    rng = random.Random(seed)
    if n_total <= 4 * max_candidates:
        combos = [dict(zip(keys, items)) for items in itertools.product(*values)]
        if len(combos) > max_candidates:
            combos = rng.sample(combos, max_candidates)
    else:  # sample without materialising a huge grid
        combos, seen = [], set()
        while len(combos) < max_candidates:
            item = tuple(rng.randrange(len(v)) for v in values)
            if item not in seen:
                seen.add(item)
                combos.append({k: v[i] for k, v, i in zip(keys, values, item)})
    out = [{k: base.get(k) for k in keys}] + combos
    unique, seen = [], set()
    for item in out:
        t = json.dumps(item, sort_keys=True, default=str)
        if t not in seen:
            seen.add(t)
            unique.append(dict(base, **item))
    return unique


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/filanet.yaml")
    ap.add_argument("--search-config", default="configs/postprocess_search.yaml")
    ap.add_argument("--cache-dir", default="artifacts/oof_cache")
    ap.add_argument("--output", default="artifacts/postprocess_search.json")
    ap.add_argument("--write-config", default="configs/filanet_tuned.yaml")
    ap.add_argument("--mode", choices=["coordinate", "random"], default=None)
    ap.add_argument("--max-candidates", type=int, default=None, help="random mode only")
    ap.add_argument("--max-images-per-fold", type=int, default=None, help="quick screening on a subset")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--memo", default=None, help="file storing evaluated candidates (resumable search)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly before this budget (rerun the same command to resume)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    search_cfg = load_yaml(args.search_config)
    mode = args.mode or search_cfg.get("mode", "coordinate")
    space = search_cfg["space"]
    base = postprocess_params(cfg)
    records, folds, _, group_mode = load_records_and_folds(cfg)
    n_folds = int(cfg["n_folds"])
    t0 = time.time()
    deadline = t0 + 3600.0 * args.max_hours if args.max_hours else None
    ev = Evaluator(records, folds, args.cache_dir, base, cfg.get("metric"), args.workers,
                   args.max_images_per_fold, int(cfg["seed"]), memo_path=args.memo, deadline=deadline)
    print(f"{mode} search over {len(space)} parameters on {len(ev.jobs)} OOF images", flush=True)
    try:
        payload, final = run_search(ev, mode, space, search_cfg, args, cfg, base, n_folds, group_mode)
    except BudgetExceeded:
        payload = {"metric": "official micro PQ", "mode": mode, "complete": False,
                   "images_evaluated": len(ev.jobs), "subset_images_per_fold": args.max_images_per_fold,
                   "distinct_candidates_evaluated": len(ev.memo), "memo": args.memo,
                   "seconds": round(time.time() - t0, 1)}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"time budget used after {len(ev.memo)} evaluated candidates: search saved in {args.memo}; "
              f"rerun the same command to continue", flush=True)
        return
    payload["seconds"] = round(time.time() - t0, 1)
    payload["candidates_restored_from_memo"] = ev.restored
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    tuned = copy.deepcopy(cfg)
    tuned["inference"].update({k: final[k] for k in space})
    tuned.setdefault("tuning", {}).update({
        "source": args.output,
        "mode": mode,
        "nested_micro_pq": payload["nested_micro_pq"],
        "selected_oof_micro_pq": payload["final_oof_micro_pq"],
        "base_oof_micro_pq": payload["base_oof_micro_pq"],
        "subset_images_per_fold": args.max_images_per_fold,
    })
    Path(args.write_config).parent.mkdir(parents=True, exist_ok=True)
    Path(args.write_config).write_text(yaml.safe_dump(tuned, sort_keys=False), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("base_oof_micro_pq", "nested_micro_pq", "final_oof_micro_pq",
                                              "final_params", "distinct_candidates_evaluated")}, indent=2, default=str))


def run_search(ev, mode, space, search_cfg, args, cfg, base, n_folds, group_mode):
    all_folds = list(range(n_folds))
    heldout = np.zeros(4)
    outer = []
    if mode == "coordinate":
        rounds = int(search_cfg.get("rounds", 2))
        for f in all_folds:
            sel = [g for g in all_folds if g != f]
            params, sel_pq, _ = coordinate_search(ev, space, rounds, sel)
            counts = ev.evaluate([params])[0]
            hs = ev.fold_sums(counts, [f])
            heldout += hs
            outer.append({"heldout_fold": f, "selection_pq": sel_pq, "heldout_pq": float(_pq(hs)),
                          "params": {k: params[k] for k in space}})
            print(f"outer fold {f}: selection PQ {sel_pq:.4f} -> held-out PQ {float(_pq(hs)):.4f}", flush=True)
        final, final_pq, path = coordinate_search(ev, space, rounds, all_folds)
    else:
        max_c = int(args.max_candidates or search_cfg.get("max_candidates", 48))
        cands = candidate_grid(space, max_c, int(cfg["seed"]), base)
        counts = ev.evaluate(cands)
        fold_sums = np.stack([[ev.fold_sums(c, [f]) for f in all_folds] for c in counts])  # [C, F, 4]
        for f in all_folds:
            sel = [g for g in all_folds if g != f]
            sel_pq = _pq(fold_sums[:, sel].sum(axis=1))
            c = int(np.argmax(sel_pq))
            heldout += fold_sums[c, f]
            outer.append({"heldout_fold": f, "selection_pq": float(sel_pq[c]),
                          "heldout_pq": float(_pq(fold_sums[c, f])), "params": {k: cands[c][k] for k in space}})
        all_pq = _pq(fold_sums.sum(axis=1))
        final, final_pq, path = cands[int(np.argmax(all_pq))], float(all_pq.max()), []

    base_pq = float(_pq(ev.fold_sums(ev.evaluate([base])[0], all_folds)))
    payload = {
        "metric": "official micro PQ",
        "mode": mode,
        "complete": True,
        "group_mode": group_mode,
        "images_evaluated": len(ev.jobs),
        "subset_images_per_fold": args.max_images_per_fold,
        "distinct_candidates_evaluated": len(ev.memo),
        "base_oof_micro_pq": base_pq,
        "nested_micro_pq": float(_pq(heldout)),
        "outer_selections": outer,
        "final_oof_micro_pq": final_pq,
        "final_params": {k: final[k] for k in space},
        "final_search_path": path,
    }
    return payload, final


if __name__ == "__main__":
    main()
