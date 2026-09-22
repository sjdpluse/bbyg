from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from truetrade.scalper.research_models import binary_metrics, fit_predict_architecture
from truetrade.scalper.sample_intervals import load_label_intervals
from truetrade.scalper.store import ScalperStore


ARCHITECTURES = ("linear", "quadratic", "mlp")


def _eligible_train_indices(rows, intervals, validation_start: int) -> np.ndarray:
    validation_start_ts = rows[validation_start].feature_ts_ns
    eligible = [
        i for i in range(validation_start)
        if rows[i].feature_ts_ns in intervals
        and intervals[rows[i].feature_ts_ns].label_end_ts_ns < validation_start_ts
    ]
    return np.asarray(eligible, dtype=int)


def _evaluate_candidate(
    architecture: str,
    x: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    val_start: int,
    val_end: int,
    *,
    recent: int,
    linear_iterations: int,
    mlp_iterations: int,
    seed: int,
) -> dict:
    if recent > 0 and len(train_indices) > recent:
        train_indices = train_indices[-recent:]
    train_x = x[train_indices]
    train_y = y[train_indices]
    val_x = x[val_start:val_end]
    val_y = y[val_start:val_end]
    p = fit_predict_architecture(
        architecture,
        train_x,
        train_y,
        val_x,
        linear_iterations=linear_iterations,
        mlp_iterations=mlp_iterations,
        seed=seed,
    )
    metrics = binary_metrics(val_y, p)
    metrics["train_samples"] = int(len(train_indices))
    metrics["train_long_fraction"] = float(np.mean(train_y))
    return metrics


def _baseline(y_train: np.ndarray, y_val: np.ndarray) -> dict:
    prior = float(np.clip(np.mean(y_train), 1e-9, 1.0 - 1e-9))
    p = np.full(len(y_val), prior, dtype=float)
    result = binary_metrics(y_val, p)
    result["train_long_fraction"] = prior
    return result


def _summarize(results: list[dict]) -> dict:
    logloss = np.asarray([float(r["logloss"]) for r in results])
    improvements = np.asarray([float(r["logloss_improvement_vs_baseline"]) for r in results])
    balanced = np.asarray([float(r["balanced_accuracy"]) for r in results])
    accuracy = np.asarray([float(r["accuracy"]) for r in results])
    brier = np.asarray([float(r["brier"]) for r in results])
    probability_std = np.asarray([float(r["probability_std"]) for r in results])
    return {
        "folds": int(len(results)),
        "mean_logloss": float(logloss.mean()),
        "median_logloss": float(np.median(logloss)),
        "mean_logloss_improvement_vs_baseline": float(improvements.mean()),
        "positive_logloss_folds": int((improvements > 0).sum()),
        "mean_accuracy": float(accuracy.mean()),
        "mean_balanced_accuracy": float(balanced.mean()),
        "median_balanced_accuracy": float(np.median(balanced)),
        "worst_balanced_accuracy": float(balanced.min()),
        "best_balanced_accuracy": float(balanced.max()),
        "mean_brier": float(brier.mean()),
        "mean_probability_std": float(probability_std.mean()),
    }


def _passes_research_gate(summary: dict, latest: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if summary["positive_logloss_folds"] < 5:
        reasons.append("fewer_than_5_of_6_positive_logloss_folds")
    if summary["mean_logloss_improvement_vs_baseline"] <= 0.0:
        reasons.append("non_positive_mean_logloss_improvement")
    if summary["mean_balanced_accuracy"] < 0.53:
        reasons.append("mean_balanced_accuracy_below_0_53")
    if summary["worst_balanced_accuracy"] < 0.49:
        reasons.append("worst_fold_balanced_accuracy_below_0_49")
    if latest["logloss_improvement_vs_baseline"] <= 0.0:
        reasons.append("latest_holdout_logloss_not_improved")
    if latest["balanced_accuracy"] < 0.52:
        reasons.append("latest_holdout_balanced_accuracy_below_0_52")
    return not reasons, reasons


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only leakage-safe nonlinear benchmark for BBYG tick learning"
    )
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--validation", type=int, default=1000)
    parser.add_argument("--latest-validation", type=int, default=120)
    parser.add_argument("--min-train", type=int, default=10000)
    parser.add_argument("--recent", type=int, default=10000)
    parser.add_argument("--linear-iterations", type=int, default=160)
    parser.add_argument("--mlp-iterations", type=int, default=180)
    parser.add_argument("--seed", type=int, default=731022)
    args = parser.parse_args()
    if args.folds < 3 or args.validation < 200 or args.latest_validation < 100:
        raise SystemExit("invalid validation settings")
    if args.min_train < 2000 or args.recent < args.min_train // 2:
        raise SystemExit("invalid training settings")

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
        raise SystemExit("label interval coverage incomplete; rebuild learning first")

    n = len(rows)
    x = np.asarray([r.sample.x for r in rows], dtype=float)
    y = np.asarray([r.sample.y for r in rows], dtype=int)
    latest_start = n - args.latest_validation
    if latest_start <= args.min_train:
        raise SystemExit("not enough samples for latest independent holdout")

    # The walk-forward folds end before the final independent latest block so that the
    # same newest evidence is not used for architecture selection and final confirmation.
    latest_fold_end = latest_start - args.validation
    earliest_fold_start = args.min_train
    if latest_fold_end <= earliest_fold_start:
        raise SystemExit("not enough samples for requested folds and final holdout")
    starts = np.unique(np.linspace(earliest_fold_start, latest_fold_end, args.folds, dtype=int))

    aggregate: dict[str, list[dict]] = {a: [] for a in ARCHITECTURES}
    folds: list[dict] = []
    for fold_no, val_start in enumerate(starts, start=1):
        val_start = int(val_start)
        val_end = val_start + args.validation
        train_indices = _eligible_train_indices(rows, intervals, val_start)
        if len(train_indices) < args.min_train:
            raise SystemExit(f"fold {fold_no}: insufficient leakage-safe training samples")
        selected = train_indices[-args.recent:] if len(train_indices) > args.recent else train_indices
        baseline = _baseline(y[selected], y[val_start:val_end])
        fold = {
            "fold": fold_no,
            "validation_start_sample_id": int(rows[val_start].sample_id),
            "validation_end_sample_id": int(rows[val_end - 1].sample_id),
            "eligible_training_samples": int(len(train_indices)),
            "purged_training_samples": int(val_start - len(train_indices)),
            "baseline": baseline,
            "candidates": {},
        }
        for arch in ARCHITECTURES:
            metrics = _evaluate_candidate(
                arch, x, y, train_indices, val_start, val_end,
                recent=args.recent,
                linear_iterations=args.linear_iterations,
                mlp_iterations=args.mlp_iterations,
                seed=args.seed + fold_no,
            )
            metrics["logloss_improvement_vs_baseline"] = float(baseline["logloss"] - metrics["logloss"])
            fold["candidates"][arch] = metrics
            aggregate[arch].append(metrics)
        folds.append(fold)

    summary = {arch: _summarize(results) for arch, results in aggregate.items()}

    latest_train = _eligible_train_indices(rows, intervals, latest_start)
    if len(latest_train) < args.min_train:
        raise SystemExit("latest holdout: insufficient leakage-safe training samples")
    latest_selected = latest_train[-args.recent:] if len(latest_train) > args.recent else latest_train
    latest_baseline = _baseline(y[latest_selected], y[latest_start:n])
    latest_candidates: dict[str, dict] = {}
    for arch in ARCHITECTURES:
        metrics = _evaluate_candidate(
            arch, x, y, latest_train, latest_start, n,
            recent=args.recent,
            linear_iterations=args.linear_iterations,
            mlp_iterations=args.mlp_iterations,
            seed=args.seed + 1000,
        )
        metrics["logloss_improvement_vs_baseline"] = float(latest_baseline["logloss"] - metrics["logloss"])
        latest_candidates[arch] = metrics

    gates = {}
    for arch in ARCHITECTURES:
        passed, reasons = _passes_research_gate(summary[arch], latest_candidates[arch])
        gates[arch] = {"passed": passed, "reasons": reasons}

    # This rank is research-only and cannot authorize trading. It exists solely to make
    # architecture comparison deterministic for the next engineering step.
    ordered = sorted(
        ARCHITECTURES,
        key=lambda a: (
            not gates[a]["passed"],
            -summary[a]["positive_logloss_folds"],
            -summary[a]["mean_logloss_improvement_vs_baseline"],
            -summary[a]["mean_balanced_accuracy"],
            latest_candidates[a]["logloss"],
        ),
    )

    print(json.dumps({
        "mode": "read_only_nonlinear_research",
        "execution_authorized": False,
        "samples": n,
        "label_interval_coverage": coverage,
        "settings": {
            "folds": len(starts),
            "validation": args.validation,
            "latest_validation": args.latest_validation,
            "min_train": args.min_train,
            "recent": args.recent,
            "linear_iterations": args.linear_iterations,
            "mlp_iterations": args.mlp_iterations,
            "seed": args.seed,
        },
        "summary": summary,
        "latest_holdout": {
            "validation_start_sample_id": int(rows[latest_start].sample_id),
            "validation_end_sample_id": int(rows[-1].sample_id),
            "purged_training_samples": int(latest_start - len(latest_train)),
            "baseline": latest_baseline,
            "candidates": latest_candidates,
        },
        "research_gate": gates,
        "research_order": ordered,
        "folds": folds,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
