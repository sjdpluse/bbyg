from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from truetrade.scalper.research_models import binary_metrics, fit_predict_architecture, restore_training_prior
from truetrade.scalper.selective import choose_selective_threshold, selective_metrics
from truetrade.scalper.store import ScalperStore


ARCHITECTURES = ("linear", "quadratic", "mlp")
VARIANTS = ("raw_balanced", "prior_corrected")
DAY_NS = 86_400_000_000_000


def _load_dataset(store: ScalperStore):
    sample_count = store.sample_count()
    interval_count = int(store.db.execute(
        "SELECT count(*) FROM sample_label_intervals"
    ).fetchone()[0])
    if sample_count == 0:
        raise SystemExit("no labeled samples")
    if interval_count != sample_count:
        raise SystemExit(
            f"label interval coverage incomplete: samples={sample_count}, intervals={interval_count}"
        )

    sample_ids = np.empty(sample_count, dtype=np.int64)
    feature_ts = np.empty(sample_count, dtype=np.int64)
    label_end_ts = np.empty(sample_count, dtype=np.int64)
    x = np.empty((sample_count, 8), dtype=float)
    y = np.empty(sample_count, dtype=np.int8)
    cursor = store.db.execute(
        """SELECT s.id,s.feature_ts_ns,s.x_json,s.y,i.label_end_ts_ns
           FROM samples s
           JOIN sample_label_intervals i ON i.feature_ts_ns=s.feature_ts_ns
           ORDER BY s.id"""
    )
    loaded = 0
    for loaded, (sample_id, ts_ns, x_json, label, end_ts) in enumerate(cursor, start=1):
        values = json.loads(x_json)
        if len(values) != 8:
            raise SystemExit(f"unexpected feature width at sample {sample_id}: {len(values)}")
        pos = loaded - 1
        sample_ids[pos] = int(sample_id)
        feature_ts[pos] = int(ts_ns)
        label_end_ts[pos] = int(end_ts)
        x[pos] = np.asarray(values, dtype=float)
        y[pos] = int(label)
    if loaded != sample_count:
        raise SystemExit(f"sample load mismatch: expected {sample_count}, loaded {loaded}")
    if not np.isfinite(x).all():
        raise SystemExit("non-finite feature detected")
    return sample_ids, feature_ts, label_end_ts, x, y


def _day_ranges(feature_ts: np.ndarray) -> list[tuple[str, int, int]]:
    day_key = feature_ts // DAY_NS
    boundaries = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.flatnonzero(np.diff(day_key) != 0).astype(np.int64) + 1,
        np.asarray([len(feature_ts)], dtype=np.int64),
    ))
    result: list[tuple[str, int, int]] = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        start, end = int(a), int(b)
        day = datetime.fromtimestamp(feature_ts[start] / 1_000_000_000, tz=timezone.utc).date().isoformat()
        result.append((day, start, end))
    return result


def _baseline(train_y: np.ndarray, validation_y: np.ndarray) -> dict:
    prior = float(np.clip(np.mean(train_y), 1e-9, 1.0 - 1e-9))
    metrics = binary_metrics(validation_y, np.full(len(validation_y), prior, dtype=float))
    metrics["train_long_fraction"] = prior
    return metrics


def _variant_predictions(
    architecture: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: np.ndarray,
    *,
    linear_iterations: int,
    mlp_iterations: int,
    seed: int,
) -> dict[str, np.ndarray]:
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


def _mean_or_none(values: list[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=float)))


def _summary(day_results: list[dict], architecture: str, variant: str) -> dict:
    forced = [d["architectures"][architecture][variant]["forced"] for d in day_results]
    selective = [d["architectures"][architecture][variant]["selective"] for d in day_results]
    forced_bal = [float(v["balanced_accuracy"]) for v in forced]
    improvements = [float(v["logloss_improvement_vs_baseline"]) for v in forced]
    valid_selective = [v for v in selective if v is not None and v["selected"] > 0]
    result = {
        "days": len(day_results),
        "forced_mean_balanced_accuracy": _mean_or_none(forced_bal),
        "forced_worst_balanced_accuracy": None if not forced_bal else float(min(forced_bal)),
        "forced_mean_logloss_improvement_vs_baseline": _mean_or_none(improvements),
        "forced_positive_logloss_days": int(sum(v > 0 for v in improvements)),
        "selective_days_with_signals": len(valid_selective),
        "selective_total_selected": int(sum(int(v["selected"]) for v in valid_selective)),
    }
    if valid_selective:
        result.update({
            "selective_mean_balanced_accuracy": _mean_or_none([
                float(v["balanced_accuracy"]) for v in valid_selective if v["balanced_accuracy"] is not None
            ]),
            "selective_worst_balanced_accuracy": (
                None if not any(v["balanced_accuracy"] is not None for v in valid_selective)
                else float(min(float(v["balanced_accuracy"]) for v in valid_selective if v["balanced_accuracy"] is not None))
            ),
            "selective_mean_accuracy": _mean_or_none([float(v["accuracy"]) for v in valid_selective]),
            "selective_mean_coverage": _mean_or_none([float(v["coverage"]) for v in valid_selective]),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only day-isolated nonlinear/selective walk-forward benchmark for BBYG"
    )
    parser.add_argument("--min-train", type=int, default=10_000)
    parser.add_argument("--recent", type=int, default=20_000)
    parser.add_argument("--calibration", type=int, default=5_000)
    parser.add_argument("--min-day-samples", type=int, default=3_000)
    parser.add_argument("--linear-iterations", type=int, default=140)
    parser.add_argument("--mlp-iterations", type=int, default=120)
    parser.add_argument("--seed", type=int, default=731022)
    args = parser.parse_args()
    if args.min_train < 2_000 or args.recent < args.min_train:
        raise SystemExit("invalid training settings")
    if args.calibration < 1_000 or args.min_day_samples < 500:
        raise SystemExit("invalid calibration/day settings")

    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-multiday-v1"))
    store = ScalperStore(state_dir / "scalper.sqlite")
    try:
        sample_ids, feature_ts, label_end_ts, x, y = _load_dataset(store)
    finally:
        store.close()

    day_ranges = _day_ranges(feature_ts)
    eligible_days: list[tuple[str, int, int]] = []
    for day, val_start, val_end in day_ranges:
        if val_end - val_start < args.min_day_samples:
            continue
        validation_start_ts = int(feature_ts[val_start])
        resolved_before_day = np.flatnonzero(label_end_ts[:val_start] < validation_start_ts)
        if len(resolved_before_day) < args.min_train + args.calibration:
            continue
        eligible_days.append((day, val_start, val_end))

    if len(eligible_days) < 2:
        raise SystemExit("need at least two eligible out-of-sample days")

    latest_day = eligible_days[-1][0]
    day_outputs: list[dict] = []
    for fold_no, (day, val_start, val_end) in enumerate(eligible_days, start=1):
        validation_start_ts = int(feature_ts[val_start])
        pre = np.flatnonzero(label_end_ts[:val_start] < validation_start_ts)
        cal_indices = pre[-args.calibration:]
        calibration_start_index = int(cal_indices[0])
        calibration_start_ts = int(feature_ts[calibration_start_index])
        train_indices = np.flatnonzero(label_end_ts[:calibration_start_index] < calibration_start_ts)
        if len(train_indices) < args.min_train:
            continue
        if len(train_indices) > args.recent:
            train_indices = train_indices[-args.recent:]

        train_x = x[train_indices]
        train_y = y[train_indices]
        validation_x = x[val_start:val_end]
        validation_y = y[val_start:val_end]
        eval_x = np.concatenate((x[cal_indices], validation_x), axis=0)
        baseline = _baseline(train_y, validation_y)

        output = {
            "fold": fold_no,
            "utc_date": day,
            "is_latest_holdout": day == latest_day,
            "train_samples": int(len(train_indices)),
            "calibration_samples": int(len(cal_indices)),
            "validation_samples": int(val_end - val_start),
            "validation_start_sample_id": int(sample_ids[val_start]),
            "validation_end_sample_id": int(sample_ids[val_end - 1]),
            "baseline": baseline,
            "architectures": {},
        }
        for arch_no, architecture in enumerate(ARCHITECTURES):
            predictions = _variant_predictions(
                architecture,
                train_x,
                train_y,
                eval_x,
                linear_iterations=args.linear_iterations,
                mlp_iterations=args.mlp_iterations,
                seed=args.seed + fold_no * 10 + arch_no,
            )
            arch_result = {}
            for variant, p in predictions.items():
                cal_p = p[:len(cal_indices)]
                val_p = p[len(cal_indices):]
                forced = binary_metrics(validation_y, val_p)
                forced["logloss_improvement_vs_baseline"] = float(
                    baseline["logloss"] - forced["logloss"]
                )
                threshold, threshold_table = choose_selective_threshold(
                    y[cal_indices],
                    cal_p,
                    min_coverage=0.05,
                    min_selected=100,
                    min_class_count=30,
                )
                selective = None
                if threshold is not None:
                    selective = selective_metrics(validation_y, val_p, threshold).__dict__
                arch_result[variant] = {
                    "forced": forced,
                    "chosen_threshold": threshold,
                    "selective": selective,
                    "calibration_candidates": threshold_table,
                }
            output["architectures"][architecture] = arch_result
        day_outputs.append(output)
        print(json.dumps({
            "stage": "day_complete",
            "utc_date": day,
            "is_latest_holdout": day == latest_day,
            "validation_samples": int(val_end - val_start),
        }, sort_keys=True), flush=True)

    research_days = [d for d in day_outputs if not d["is_latest_holdout"]]
    holdout = next(d for d in day_outputs if d["is_latest_holdout"])
    summary = {
        architecture: {
            variant: _summary(research_days, architecture, variant)
            for variant in VARIANTS
        }
        for architecture in ARCHITECTURES
    }

    print(json.dumps({
        "stage": "complete",
        "mode": "read_only_day_isolated_research",
        "execution_authorized": False,
        "state_dir": str(state_dir),
        "samples": int(len(y)),
        "label_interval_coverage": 1.0,
        "settings": {
            "min_train": args.min_train,
            "recent": args.recent,
            "calibration": args.calibration,
            "min_day_samples": args.min_day_samples,
            "linear_iterations": args.linear_iterations,
            "mlp_iterations": args.mlp_iterations,
        },
        "eligible_dates": [d[0] for d in eligible_days],
        "research_dates": [d["utc_date"] for d in research_days],
        "latest_holdout_date": holdout["utc_date"],
        "summary": summary,
        "latest_holdout": holdout,
        "days": day_outputs,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
