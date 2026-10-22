"""Train and evaluate the learned instance rescoring on the cached OOF predictions.

Every predicted filament of every OOF image is described by features
(``src/rescore.py``: prediction maps, and optionally the H-alpha image itself)
and labelled with its exact contribution to the official micro PQ. A
gradient-boosted regressor predicts that contribution; instances below a
threshold are dropped.

Several **variants** are compared, each a candidate set x a feature set:

* ``filtered/maps``: the instances of the calibrated post-processing, map features;
* ``filtered/maps+image``: same instances, plus features of the grayscale image;
* ``permissive/maps+image``: candidates produced without the calibrated
  confidence/area filters (``min_instance_score`` 0, ``min_instance_area``
  ``--permissive-area``), so that the learned filter alone decides and true
  filaments discarded by the crude filters can be recovered.

Honest evaluation (nested, by fold): for each outer fold ``k``

* the reconstruction setting is the one the calibration chose WITHOUT fold
  ``k`` (``--search-json``, ``outer_selections``);
* for every variant, the threshold comes from an inner cross-validation on the
  other folds; the variant with the best inner PQ is chosen, its model is
  fitted on the other folds, and fold ``k`` is scored once.

The final variant is the one with the best cross-fitted PQ on all folds; its
model (all folds) and threshold are saved with joblib and enabled in
``--write-config`` only when the nested PQ with rescoring beats the nested PQ
of the calibrated post-processing alone on the same images.

Feature extraction (one reconstruction per image and setting) is the slow
part; it is cached per image under ``--output-dir/cache`` and resumable
(``--max-hours``): rerun the same command to continue.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # repo root, so `src` imports work

import argparse
import copy
import hashlib
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.engine import load_records_and_folds, postprocess_params
from src.evaluation import group_records_by_image, label_map_to_rles, score_image
from src.oof import RECONSTRUCTION_KEYS, cache_path, load_prediction
from src.oof_eval import default_workers
from src.reconstruct import reconstruct_label_map
from src.rescore import (FEATURE_SETS, FEATURES_VERSION, best_threshold, fit_model, image_features,
                         instance_contributions, instance_features, pq_of_selection, target)
from src.utils import load_yaml

CANDIDATE_SETS = ("filtered", "permissive")
VARIANTS = (("filtered", "maps"), ("filtered", "maps+image"), ("permissive", "maps+image"))


def _token(params) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _oof_signature(cache_dir, metric_cfg) -> str:
    shas = {}
    for m in sorted(Path(cache_dir).glob("manifest*.json")):
        try:
            for fold, info in json.loads(m.read_text()).get("folds", {}).items():
                shas[str(fold)] = info.get("weights_sha256")
        except (OSError, ValueError):
            continue
    blob = json.dumps({"weights": shas, "features": FEATURES_VERSION,
                       "iou": float((metric_cfg or {}).get("iou_match_threshold", 0.5))}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _extract(job):
    pred_path, image_path, recs, todo, threshold = job
    pred = load_prediction(pred_path, keys=RECONSTRUCTION_KEYS)
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image_path)
    for params, out_path in todo:
        labels, _ = reconstruct_label_map(
            pred["region"], pred["centerline"], pred["boundary"], pred["distance"],
            orientation=pred.get("orientation"), width=pred.get("width"),
            instance_embedding=pred.get("instance_embedding"), **params,
        )
        x = instance_features(labels, pred)
        xi = image_features(labels, gray)
        scores = score_image(recs, label_map_to_rles(labels), threshold)
        a, d, g = instance_contributions(scores, int(labels.max()))
        base = [(s["iou_sum"], s["tp"], s["fp"], s["fn"]) for s in scores if s["scored"]]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(out_path) + ".tmp.npz")
        np.savez(tmp, x=x, xi=xi, a=a, d=d, g=np.int64(g), base=np.asarray(base, np.float64).reshape(-1, 4))
        os.replace(tmp, out_path)
    return len(todo)


def extract_all(records, folds, cache_dir, image_dir, param_sets, out_root, workers, threshold, deadline):
    """Cache the instances of every OOF image under every setting; returns True when complete."""
    by_file = group_records_by_image(records)
    jobs = []
    for file_name, idx in sorted(by_file.items()):
        fold = int(folds[idx[0]])
        todo = []
        for params in param_sets:
            out = Path(out_root) / _token(params) / f"fold{fold}_{Path(file_name).stem}.npz"
            if not out.exists():
                todo.append((params, str(out)))
        if todo:
            jobs.append((str(cache_path(cache_dir, fold, file_name)), str(Path(image_dir) / file_name),
                         [records[i] for i in idx], todo, threshold))
    total = len(jobs)
    print(f"feature extraction: {total} image(s) to process, {len(param_sets)} setting(s)", flush=True)
    if not jobs:
        return True
    done, pending, it = 0, set(), iter(jobs)
    stopped = False
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as ex:
        while True:
            while not stopped and len(pending) < 2 * max(1, int(workers)):
                if deadline is not None and time.time() > deadline:
                    stopped = True
                    break
                job = next(it, None)
                if job is None:
                    break
                pending.add(ex.submit(_extract, job))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                f.result()
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"  extracted {done}/{total} images", flush=True)
    return done == total


def load_set(records, folds, params, out_root):
    by_file = group_records_by_image(records)
    xs, xis, a, d, inst_fold, base = [], [], [], [], [], []
    g_by_fold = {}
    for file_name, idx in sorted(by_file.items()):
        fold = int(folds[idx[0]])
        with np.load(Path(out_root) / _token(params) / f"fold{fold}_{Path(file_name).stem}.npz") as z:
            xs.append(z["x"])
            xis.append(z["xi"])
            a.append(z["a"])
            d.append(z["d"])
            inst_fold.append(np.full(len(z["a"]), fold, np.int64))
            g_by_fold[fold] = g_by_fold.get(fold, 0) + int(z["g"])
            base.append(np.c_[np.full(len(z["base"]), fold), z["base"]])
    x = np.concatenate(xs).astype(np.float64)
    xi = np.concatenate(xis).astype(np.float64)
    return {"x": {"maps": x, "maps+image": np.hstack([x, xi])}, "a": np.concatenate(a), "d": np.concatenate(d),
            "fold": np.concatenate(inst_fold), "g": g_by_fold, "base": np.concatenate(base)}


def _g(data, fold_list):
    return sum(data["g"].get(f, 0) for f in fold_list)


def cross_fit(data, features, fold_list, seed):
    """Predictions for the instances of ``fold_list``, each fold by a model fitted on the others."""
    x = data["x"][features]
    y = target(data["a"], data["d"])
    out = np.full(len(y), np.nan)
    for f in fold_list:
        train = np.isin(data["fold"], [g for g in fold_list if g != f])
        test = data["fold"] == f
        if test.any():
            # no training fold (2-fold smoke runs): constant prediction, i.e. keep every instance
            out[test] = fit_model(x[train], y[train], seed).predict(x[test]) if train.any() else 1.0
    return out


def _sums(a, d, g, keep):
    return np.array([a[keep].sum(), 0.5 * g + d[keep].sum()])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="config whose post-processing produces the instances (tuned)")
    ap.add_argument("--search-json", default=None,
                    help="postprocess_search.json: per-outer-fold settings for a fully nested estimate")
    ap.add_argument("--cache-dir", default="artifacts/oof_cache")
    ap.add_argument("--output-dir", default="artifacts/rescoring")
    ap.add_argument("--write-config", default=None, help="write the config with the rescoring section here")
    ap.add_argument("--permissive-area", type=int, default=100,
                    help="min_instance_area of the permissive candidate set")
    ap.add_argument("--variants", nargs="+", default=[f"{c}/{f}" for c, f in VARIANTS],
                    help="candidate/features variants to compare")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    variants = [tuple(v.split("/", 1)) for v in args.variants]
    for c, f in variants:
        if c not in CANDIDATE_SETS or f not in FEATURE_SETS:
            raise SystemExit(f"unknown variant {c}/{f}")
    overrides = {"filtered": {}, "permissive": {"min_instance_score": 0.0, "min_instance_area": int(args.permissive_area)}}

    t0 = time.time()
    deadline = t0 + 3600.0 * args.max_hours if args.max_hours else None
    cfg = load_yaml(args.config)
    records, folds, image_dir, group_mode = load_records_and_folds(cfg)
    fold_ids = sorted(set(int(f) for f in folds))
    final_params = postprocess_params(cfg)
    outer_params, nested_settings = {}, False
    if args.search_json and Path(args.search_json).exists():
        search = json.loads(Path(args.search_json).read_text())
        if search.get("complete"):
            for sel in search.get("outer_selections", []):
                outer_params[int(sel["heldout_fold"])] = dict(final_params, **sel["params"])
            nested_settings = set(outer_params) == set(fold_ids)
    if not nested_settings:
        outer_params = {f: final_params for f in fold_ids}

    def setting(base, cand):
        return dict(base, **overrides[cand])

    cands = sorted({"filtered"} | {c for c, _ in variants})
    param_sets, seen = [], set()
    for base in [final_params, *outer_params.values()]:
        for c in cands:
            p = setting(base, c)
            if _token(p) not in seen:
                seen.add(_token(p))
                param_sets.append(p)

    out_dir = Path(args.output_dir)
    root = out_dir / "cache" / _oof_signature(args.cache_dir, cfg.get("metric"))
    threshold = float(cfg.get("metric", {}).get("iou_match_threshold", 0.5))
    complete = extract_all(records, folds, args.cache_dir, image_dir, param_sets, root, args.workers, threshold,
                           deadline)
    report_path = out_dir / "rescoring.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not complete:
        report_path.write_text(json.dumps({"complete": False, "seconds": round(time.time() - t0, 1)}, indent=2))
        print("time budget used: features kept in", root, "- rerun the same command to continue", flush=True)
        return

    loaded = {}

    def data_for(params):
        key = _token(params)
        if key not in loaded:
            loaded[key] = load_set(records, folds, params, root)
        return loaded[key]

    # sanity: keeping every instance reproduces the official counts exactly
    for params in param_sets:
        data = data_for(params)
        s = data["base"][:, 1:].sum(axis=0)
        official = s[0] / (s[1] + 0.5 * s[2] + 0.5 * s[3])
        mine = pq_of_selection(data["a"], data["d"], _g(data, fold_ids), np.ones(len(data["a"]), bool))
        if abs(official - mine) > 1e-9:
            raise AssertionError(f"setting {_token(params)}: separable PQ {mine} != official {official}")

    # nested estimate: per outer fold, the variant with the best inner-CV PQ on the other folds
    names = [f"{c}/{f}" for c, f in variants]
    sum_res, sum_base, per_fold = np.zeros(2), np.zeros(2), []
    per_variant = {n: np.zeros(2) for n in names}
    for k in fold_ids:
        others = [f for f in fold_ids if f != k]
        base_data = data_for(setting(outer_params[k], "filtered"))
        held_b = base_data["fold"] == k
        g_k = _g(base_data, [k])
        base = _sums(base_data["a"][held_b], base_data["d"][held_b], g_k, np.ones(int(held_b.sum()), bool))
        options = []
        for (c, f), name in zip(variants, names):
            data = data_for(setting(outer_params[k], c))
            y = target(data["a"], data["d"])
            sel = np.isin(data["fold"], others)
            inner = cross_fit(data, f, others, args.seed)
            thr, sel_pq = best_threshold(inner[sel], data["a"][sel], data["d"][sel], _g(data, others))
            held = data["fold"] == k
            x = data["x"][f]
            pred_k = fit_model(x[sel], y[sel], args.seed).predict(x[held]) if sel.any() else np.ones(int(held.sum()))
            keep = pred_k >= thr
            res = _sums(data["a"][held], data["d"][held], g_k, keep)
            per_variant[name] += res
            options.append({"variant": name, "threshold": thr, "selection_pq": sel_pq,
                            "heldout_pq": float(res[0] / res[1]), "kept": int(keep.sum()),
                            "candidates": int(held.sum()), "sums": res})
        best = max(options, key=lambda o: o["selection_pq"])
        sum_res += best["sums"]
        sum_base += base
        per_fold.append({"heldout_fold": k, "chosen_variant": best["variant"], "threshold": best["threshold"],
                         "selection_pq": best["selection_pq"], "heldout_pq_without": float(base[0] / base[1]),
                         "heldout_pq_with": best["heldout_pq"], "kept": best["kept"], "candidates": best["candidates"],
                         "variants": [{k2: v for k2, v in o.items() if k2 != "sums"} for o in options]})
        print(f"outer fold {k}: {best['variant']} (inner PQ {best['selection_pq']:.4f}) | held-out PQ "
              f"{base[0] / base[1]:.4f} -> {best['heldout_pq']:.4f} (kept {best['kept']} of {best['candidates']}) | "
              + ", ".join(f"{o['variant']} {o['heldout_pq']:.4f}" for o in options), flush=True)
    nested_with, nested_without = float(sum_res[0] / sum_res[1]), float(sum_base[0] / sum_base[1])
    nested_by_variant = {n: float(v[0] / v[1]) for n, v in per_variant.items()}

    # final variant: best cross-fitted PQ on all folds with the tuned setting
    finals = []
    for (c, f), name in zip(variants, names):
        data = data_for(setting(final_params, c))
        cf = cross_fit(data, f, fold_ids, args.seed)
        thr, cf_pq = best_threshold(cf, data["a"], data["d"], _g(data, fold_ids))
        finals.append((cf_pq, name, c, f, thr))
    cf_pq, name, cand, feats, thr = max(finals, key=lambda t: t[0])
    data = data_for(setting(final_params, cand))
    y = target(data["a"], data["d"])
    _, oracle_pq = best_threshold(y, data["a"], data["d"], _g(data, fold_ids))
    base_final = data_for(setting(final_params, "filtered"))
    model = fit_model(data["x"][feats], y, args.seed)
    import joblib
    import sklearn

    model_path = out_dir / "rescorer.joblib"
    joblib.dump({"model": model, "feature_set": feats, "feature_names": FEATURE_SETS[feats],
                 "features_version": FEATURES_VERSION, "candidate_overrides": overrides[cand], "threshold": thr,
                 "sklearn": sklearn.__version__, "params_token": _token(final_params)}, model_path)
    enabled = nested_with > nested_without
    report = {
        "complete": True,
        "metric": "official micro PQ (separable per instance; checked against the official counts)",
        "group_mode": group_mode,
        "images": int(len(group_records_by_image(records))),
        "reconstruction_settings_nested": nested_settings,
        "nested_micro_pq_without_rescoring": nested_without,
        "nested_micro_pq_with_rescoring": nested_with,
        "nested_micro_pq_by_variant": nested_by_variant,
        "enabled": enabled,
        "outer_folds": per_fold,
        "final_variant": name,
        "final_candidate_overrides": overrides[cand],
        "final_threshold": thr,
        "cross_fitted_micro_pq_by_variant": {n: p for p, n, *_ in finals},
        "all_instances_micro_pq": pq_of_selection(base_final["a"], base_final["d"], _g(base_final, fold_ids),
                                                  np.ones(len(base_final["a"]), bool)),
        "oracle_filtering_ceiling_micro_pq": oracle_pq,
        "instances": int(len(y)),
        "features": list(FEATURE_SETS[feats]),
        "model": str(model_path),
        "sklearn": sklearn.__version__,
        "seconds": round(time.time() - t0, 1),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("nested_micro_pq_without_rescoring", "nested_micro_pq_with_rescoring",
                                             "nested_micro_pq_by_variant", "enabled", "final_variant",
                                             "final_threshold", "oracle_filtering_ceiling_micro_pq",
                                             "reconstruction_settings_nested")}, indent=2), flush=True)
    if not nested_settings:
        print("note: the reconstruction setting was not chosen per outer fold (no complete --search-json): "
              "the nested estimate covers the rescoring only", flush=True)

    if args.write_config:
        tuned = copy.deepcopy(cfg)
        try:
            model_ref = os.path.relpath(model_path.resolve(), Path.cwd())
            if model_ref.startswith(".."):
                model_ref = str(model_path.resolve())
        except ValueError:
            model_ref = str(model_path.resolve())
        tuned["rescoring"] = {"enabled": bool(enabled), "model": model_ref, "threshold": float(thr),
                              "feature_set": feats, "candidate_overrides": overrides[cand],
                              "features_version": FEATURES_VERSION, "variant": name,
                              "nested_micro_pq_with": nested_with, "nested_micro_pq_without": nested_without,
                              "source": str(report_path)}
        Path(args.write_config).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_config).write_text(yaml.safe_dump(tuned, sort_keys=False), encoding="utf-8")
        print((f"rescoring ENABLED ({name}) in " if enabled else "rescoring not better on held-out folds: disabled in ")
              + str(args.write_config), flush=True)


if __name__ == "__main__":
    main()
