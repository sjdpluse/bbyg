from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from truetrade.scalper.store import ScalperStore


EPS = 1e-9


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _robust_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(x, axis=0)
    q25 = np.quantile(x, 0.25, axis=0)
    q75 = np.quantile(x, 0.75, axis=0)
    scale = (q75 - q25) / 1.349
    std = np.std(x, axis=0)
    scale = np.where(scale > 1e-6, scale, np.where(std > 1e-6, std, 1.0))
    return median, scale


def _robust_apply(x: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((x - median) / scale, -8.0, 8.0)


def _fit_logit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    l2: float,
    class_balanced: bool,
) -> tuple[np.ndarray, float]:
    w = np.zeros(x.shape[1], dtype=float)
    b = 0.0
    if class_balanced:
        n = len(y)
        n1 = max(int(y.sum()), 1)
        n0 = max(n - n1, 1)
        weights = np.where(y == 1, n / (2.0 * n1), n / (2.0 * n0))
    else:
        weights = np.ones(len(y), dtype=float)
    weight_sum = float(weights.sum())

    for _ in range(iterations):
        p = _sigmoid(x @ w + b)
        err = (p - y) * weights
        grad_w = (x.T @ err) / weight_sum + l2 * w
        grad_b = float(err.sum() / weight_sum)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        w = np.clip(w, -8.0, 8.0)
        b = float(np.clip(b, -8.0, 8.0))
    return w, b


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, EPS, 1.0 - EPS)
    pred = (p >= 0.5).astype(int)
    loss = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    acc = float(np.mean(pred == y))
    recalls = []
    for cls in (0, 1):
        mask = y == cls
        if mask.any():
            recalls.append(float(np.mean(pred[mask] == cls)))
    balanced = float(np.mean(recalls)) if recalls else 0.0
    return {
        "logloss": loss,
        "accuracy": acc,
        "balanced_accuracy": balanced,
        "actual_long_fraction": float(np.mean(y)),
        "predicted_long_probability": float(np.mean(p)),
        "calibration_bias": float(np.mean(p) - np.mean(y)),
    }


def _candidate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    recent: int | None,
    balanced: bool,
    iterations: int,
    learning_rate: float,
    l2: float,
) -> dict:
    if recent is not None:
        train_x = train_x[-recent:]
        train_y = train_y[-recent:]
    median, scale = _robust_fit(train_x)
    tx = _robust_apply(train_x, median, scale)
    vx = _robust_apply(val_x, median, scale)
    w, b = _fit_logit(
        tx,
        train_y,
        iterations=iterations,
        learning_rate=learning_rate,
        l2=l2,
        class_balanced=balanced,
    )
    result = _metrics(val_y, _sigmoid(vx @ w + b))
    result["train_samples"] = int(len(train_y))
    result["train_long_fraction"] = float(np.mean(train_y))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only chronological walk-forward benchmark for BBYG learning samples"
    )
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--validation", type=int, default=1000)
    parser.add_argument("--purge", type=int, default=40)
    parser.add_argument("--min-train", type=int, default=10000)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=1e-3)
    args = parser.parse_args()
    if args.folds < 3 or args.validation < 200 or args.purge < 1 or args.min_train < 2000:
        raise SystemExit("invalid walk-forward settings")

    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-demo"))
    store = ScalperStore(state_dir / "scalper.sqlite")
    try:
        rows = store.samples()
    finally:
        store.close()
    if len(rows) < args.min_train + args.purge + args.validation * args.folds:
        raise SystemExit("not enough labeled samples for requested walk-forward benchmark")

    x = np.asarray([r.sample.x for r in rows], dtype=float)
    y = np.asarray([r.sample.y for r in rows], dtype=int)
    n = len(rows)

    earliest = args.min_train + args.purge
    latest = n - args.validation
    starts = np.linspace(earliest, latest, args.folds, dtype=int)
    starts = np.unique(starts)

    candidates = {
        "global_balanced": (None, True),
        "global_unbalanced": (None, False),
        "recent_5000_balanced": (5000, True),
        "recent_10000_balanced": (10000, True),
        "recent_20000_balanced": (20000, True),
    }
    fold_results = []
    aggregate: dict[str, list[dict]] = {name: [] for name in candidates}

    for fold_no, val_start in enumerate(starts, start=1):
        train_end = int(val_start) - args.purge
        val_end = int(val_start) + args.validation
        train_x = x[:train_end]
        train_y = y[:train_end]
        val_x = x[val_start:val_end]
        val_y = y[val_start:val_end]

        train_prior = float(np.mean(train_y))
        baseline_p = np.full(len(val_y), np.clip(train_prior, EPS, 1.0 - EPS))
        baseline = _metrics(val_y, baseline_p)
        result = {
            "fold": fold_no,
            "train_end_sample_id": int(rows[train_end - 1].sample_id),
            "validation_start_sample_id": int(rows[val_start].sample_id),
            "validation_end_sample_id": int(rows[val_end - 1].sample_id),
            "baseline": baseline,
            "candidates": {},
        }
        for name, (recent, balanced) in candidates.items():
            metrics = _candidate(
                train_x,
                train_y,
                val_x,
                val_y,
                recent=recent,
                balanced=balanced,
                iterations=args.iterations,
                learning_rate=args.learning_rate,
                l2=args.l2,
            )
            metrics["logloss_improvement_vs_baseline"] = baseline["logloss"] - metrics["logloss"]
            result["candidates"][name] = metrics
            aggregate[name].append(metrics)
        fold_results.append(result)

    summary = {}
    for name, results in aggregate.items():
        losses = np.asarray([r["logloss"] for r in results])
        improvements = np.asarray([r["logloss_improvement_vs_baseline"] for r in results])
        balanced = np.asarray([r["balanced_accuracy"] for r in results])
        accuracy = np.asarray([r["accuracy"] for r in results])
        summary[name] = {
            "folds": len(results),
            "mean_logloss": float(losses.mean()),
            "median_logloss": float(np.median(losses)),
            "mean_logloss_improvement_vs_baseline": float(improvements.mean()),
            "positive_logloss_folds": int((improvements > 0).sum()),
            "mean_accuracy": float(accuracy.mean()),
            "mean_balanced_accuracy": float(balanced.mean()),
            "median_balanced_accuracy": float(np.median(balanced)),
            "worst_balanced_accuracy": float(balanced.min()),
            "best_balanced_accuracy": float(balanced.max()),
        }

    print(json.dumps({
        "mode": "read_only_walk_forward",
        "samples": n,
        "settings": {
            "folds": len(starts),
            "validation": args.validation,
            "purge": args.purge,
            "min_train": args.min_train,
            "iterations": args.iterations,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
        },
        "summary": summary,
        "folds": fold_results,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
