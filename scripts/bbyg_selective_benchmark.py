from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from truetrade.scalper.research_models import fit_predict_architecture, restore_training_prior
from truetrade.scalper.sample_intervals import load_label_intervals
from truetrade.scalper.selective import choose_selective_threshold, selective_metrics
from truetrade.scalper.store import ScalperStore


ARCHITECTURES = ("linear", "quadratic", "mlp")
VARIANTS = ("raw_balanced", "prior_corrected")


def _eligible_indices_before(rows, intervals, stop_index: int, resolution_before_ts: int) -> np.ndarray:
    eligible = [
        i for i in range(stop_index)
        if rows[i].feature_ts_ns in intervals
        and intervals[rows[i].feature_ts_ns].label_end_ts_ns < resolution_before_ts
    ]
    return np.asarray(eligible, dtype=int)


def _first_feasible_validation_start(rows, intervals, preferred: int, latest: int,
                                     min_train: int, calibration: int) -> int | None:
    start = max(int(preferred), int(min_train + calibration))
    while start <= latest:
        calibration_start = start - calibration
        calibration_start_ts = rows[calibration_start].feature_ts_ns
        train = _eligible_indices_before(rows, intervals, calibration_start, calibration_start_ts)
        if len(train) >= min_train:
            return start
        start += max(1, min_train - len(train))
    return None


def _calibration_indices(rows, intervals, calibration_start: int, validation_start: int) -> np.ndarray:
    validation_start_ts = rows[validation_start].feature_ts_ns
    return np.asarray([
        i for i in range(calibration_start, validation_start)
        if rows[i].feature_ts_ns in intervals
        and intervals[rows[i].feature_ts_ns].label_end_ts_ns < validation_start_ts
    ], dtype=int)


def _predict_variants(architecture: str, train_x: np.ndarray, train_y: np.ndarray,
                      eval_x: np.ndarray, *, linear_iterations: int,
                      mlp_iterations: int, seed: int) -> dict[str, np.ndarray]:
    raw = fit_predict_architecture(
        architecture,
        train_x,
        train_y,
        eval_x,
        linear_iterations=linear_iterations,
        mlp_iterations=mlp_iterations,
        seed=seed,
        restore_prior=False,
    )
    return {
        "raw_balanced": raw,
        "prior_corrected": restore_training_prior(raw, train_y),
    }


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"folds": 0}
    eligible = [r for r in rows if r["validation"]["selected"] > 0]
    if not eligible:
        return {"folds": len(rows), "folds_with_signals": 0}
    accuracies = np.asarray([r["validation"]["accuracy"] for r in eligible], dtype=float)
    balanced_values = [r["validation"]["balanced_accuracy"] for r in eligible
                       if r["validation"]["balanced_accuracy"] is not None]
    coverages = np.asarray([r["validation"]["coverage"] for r in eligible], dtype=float)
    selected = np.asarray([r["validation"]["selected"] for r in eligible], dtype=int)
    result = {
        "folds": len(rows),
        "folds_with_signals": len(eligible),
        "mean_accuracy": float(accuracies.mean()),
        "worst_accuracy": float(accuracies.min()),
        "mean_coverage": float(coverages.mean()),
        "total_selected": int(selected.sum()),
        "thresholds": [float(r["chosen_threshold"]) for r in eligible],
    }
    if balanced_values:
        balanced = np.asarray(balanced_values, dtype=float)
        result.update({
            "folds_with_two_classes": len(balanced_values),
            "mean_balanced_accuracy": float(balanced.mean()),
            "worst_balanced_accuracy": float(balanced.min()),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-safe selective/abstention research for BBYG nonlinear scores"
    )
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--validation", type=int, default=1000)
    parser.add_argument("--calibration", type=int, default=2000)
    parser.add_argument("--latest-validation", type=int, default=120)
    parser.add_argument("--min-train", type=int, default=10000)
    parser.add_argument("--recent", type=int, default=10000)
    parser.add_argument("--linear-iterations", type=int, default=160)
    parser.add_argument("--mlp-iterations", type=int, default=180)
    parser.add_argument("--seed", type=int, default=731022)
    args = parser.parse_args()
    if args.folds < 3 or args.validation < 200 or args.calibration < 500:
        raise SystemExit("invalid fold settings")
    if args.latest_validation < 100 or args.min_train < 2000 or args.recent < 2000:
        raise SystemExit("invalid sample settings")

    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-demo"))
    store = ScalperStore(state_dir / "scalper.sqlite")
    try:
        rows = store.samples()
        intervals = load_label_intervals(store)
    finally:
        store.close()
    if not rows:
        raise SystemExit("no labeled samples")
    coverage = sum(r.feature_ts_ns in intervals for r in rows) / len(rows)
    if coverage < 0.999:
        raise SystemExit("label interval coverage incomplete")

    x = np.asarray([r.sample.x for r in rows], dtype=float)
    y = np.asarray([r.sample.y for r in rows], dtype=int)
    n = len(rows)
    latest_start = n - args.latest_validation
    latest_fold_start = latest_start - args.validation
    preferred_first = args.min_train + args.calibration
    first = _first_feasible_validation_start(
        rows, intervals, preferred_first, latest_fold_start, args.min_train, args.calibration
    )
    if first is None:
        raise SystemExit("no feasible leakage-safe selective fold")
    starts = np.unique(np.linspace(first, latest_fold_start, args.folds, dtype=int))
    if len(starts) < args.folds:
        raise SystemExit("not enough distinct folds")

    aggregate: dict[str, dict[str, list[dict]]] = {
        arch: {variant: [] for variant in VARIANTS} for arch in ARCHITECTURES
    }
    fold_output: list[dict] = []

    def evaluate_boundary(validation_start: int, validation_end: int, *, seed: int) -> dict:
        calibration_start = validation_start - args.calibration
        calibration_start_ts = rows[calibration_start].feature_ts_ns
        train_indices = _eligible_indices_before(
            rows, intervals, calibration_start, calibration_start_ts
        )
        if len(train_indices) < args.min_train:
            raise SystemExit("insufficient leakage-safe train rows")
        if len(train_indices) > args.recent:
            train_indices = train_indices[-args.recent:]
        cal_indices = _calibration_indices(rows, intervals, calibration_start, validation_start)
        if len(cal_indices) < 500:
            raise SystemExit("insufficient resolved calibration rows")

        # Predict the complete calibration+validation span once from a model fit only on
        # earlier rows. Threshold selection then sees only resolved calibration labels.
        eval_start = calibration_start
        eval_x = x[eval_start:validation_end]
        result = {
            "train_samples": int(len(train_indices)),
            "calibration_rows": int(len(cal_indices)),
            "purged_calibration_rows": int(args.calibration - len(cal_indices)),
            "architectures": {},
        }
        cal_offsets = cal_indices - eval_start
        val_offsets = np.arange(validation_start, validation_end, dtype=int) - eval_start
        for arch in ARCHITECTURES:
            variants = _predict_variants(
                arch,
                x[train_indices],
                y[train_indices],
                eval_x,
                linear_iterations=args.linear_iterations,
                mlp_iterations=args.mlp_iterations,
                seed=seed,
            )
            arch_result = {}
            for variant, probabilities in variants.items():
                cal_p = probabilities[cal_offsets]
                threshold, threshold_table = choose_selective_threshold(
                    y[cal_indices],
                    cal_p,
                    min_coverage=0.05,
                    min_selected=50,
                    min_class_count=20,
                )
                if threshold is None:
                    validation_result = None
                else:
                    validation_result = selective_metrics(
                        y[validation_start:validation_end],
                        probabilities[val_offsets],
                        threshold,
                    ).__dict__
                arch_result[variant] = {
                    "chosen_threshold": threshold,
                    "calibration_candidates": threshold_table,
                    "validation": validation_result,
                }
            result["architectures"][arch] = arch_result
        return result

    for fold_no, val_start in enumerate(starts, start=1):
        val_start = int(val_start)
        boundary = evaluate_boundary(val_start, val_start + args.validation, seed=args.seed + fold_no)
        fold = {
            "fold": fold_no,
            "validation_start_sample_id": int(rows[val_start].sample_id),
            "validation_end_sample_id": int(rows[val_start + args.validation - 1].sample_id),
            **boundary,
        }
        for arch in ARCHITECTURES:
            for variant in VARIANTS:
                item = boundary["architectures"][arch][variant]
                if item["chosen_threshold"] is not None and item["validation"] is not None:
                    aggregate[arch][variant].append(item)
        fold_output.append(fold)

    # The newest 120 rows remain untouched by fold construction. Its threshold is selected
    # from the immediately preceding resolved calibration block, never from holdout labels.
    latest = evaluate_boundary(latest_start, n, seed=args.seed + 1000)
    latest_output = {
        "validation_start_sample_id": int(rows[latest_start].sample_id),
        "validation_end_sample_id": int(rows[-1].sample_id),
        **latest,
    }

    summary = {
        arch: {variant: _summary(aggregate[arch][variant]) for variant in VARIANTS}
        for arch in ARCHITECTURES
    }

    print(json.dumps({
        "mode": "read_only_selective_research",
        "execution_authorized": False,
        "samples": n,
        "label_interval_coverage": coverage,
        "settings": {
            "folds": len(starts),
            "validation": args.validation,
            "calibration": args.calibration,
            "latest_validation": args.latest_validation,
            "min_train": args.min_train,
            "recent": args.recent,
            "actual_first_fold_start": int(first),
            "threshold_grid": [0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.60],
        },
        "summary": summary,
        "latest_holdout": latest_output,
        "folds": fold_output,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
