from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from truetrade.scalper.economic_replay import (
    CostScenario,
    EconomicPolicy,
    aggregate_economic_metrics,
    economic_metrics,
    simulate_selective_trades,
)
from truetrade.scalper.labels import LabelSettings
from truetrade.scalper.research_models import fit_predict_architecture
from truetrade.scalper.selective import choose_selective_threshold
from truetrade.scalper.store import ScalperStore


DAY_NS = 86_400_000_000_000


def _load_samples(store: ScalperStore):
    count = store.sample_count()
    interval_count = int(store.db.execute("SELECT count(*) FROM sample_label_intervals").fetchone()[0])
    if count == 0 or interval_count != count:
        raise SystemExit(f"incomplete sample dataset: samples={count}, intervals={interval_count}")
    sample_ids = np.empty(count, dtype=np.int64)
    feature_ts = np.empty(count, dtype=np.int64)
    label_end_ts = np.empty(count, dtype=np.int64)
    x = np.empty((count, 8), dtype=float)
    y = np.empty(count, dtype=np.int8)
    cursor = store.db.execute(
        """SELECT s.id,s.feature_ts_ns,s.x_json,s.y,i.label_end_ts_ns
           FROM samples s JOIN sample_label_intervals i
           ON i.feature_ts_ns=s.feature_ts_ns ORDER BY s.id"""
    )
    loaded = 0
    for loaded, (sid, ts_ns, x_json, label, end_ts) in enumerate(cursor, start=1):
        values = json.loads(x_json)
        if len(values) != 8:
            raise SystemExit(f"unexpected feature width at sample {sid}: {len(values)}")
        pos = loaded - 1
        sample_ids[pos] = int(sid)
        feature_ts[pos] = int(ts_ns)
        label_end_ts[pos] = int(end_ts)
        x[pos] = np.asarray(values, dtype=float)
        y[pos] = int(label)
    if loaded != count or not np.isfinite(x).all():
        raise SystemExit("sample load failed")
    return sample_ids, feature_ts, label_end_ts, x, y


def _day_ranges(feature_ts: np.ndarray) -> list[tuple[str, int, int]]:
    keys = feature_ts // DAY_NS
    boundaries = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.flatnonzero(np.diff(keys) != 0).astype(np.int64) + 1,
        np.asarray([len(feature_ts)], dtype=np.int64),
    ))
    result = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        a, b = int(left), int(right)
        day = datetime.fromtimestamp(feature_ts[a] / 1_000_000_000, tz=timezone.utc).date().isoformat()
        result.append((day, a, b))
    return result


def _ticks_for_day(store: ScalperStore, feature_start: int, feature_end: int):
    # Include a small buffer before the first signal so searchsorted(next tick) remains
    # causal, but never include a tick from the following UTC date.
    day_start = (int(feature_start) // DAY_NS) * DAY_NS
    day_end = day_start + DAY_NS
    rows = list(store.db.execute(
        "SELECT ts_ns,bid,ask FROM ticks WHERE ts_ns>=? AND ts_ns<? ORDER BY ts_ns",
        (day_start, day_end),
    ))
    if len(rows) < 2:
        raise SystemExit("insufficient raw ticks for validation day")
    ts = np.asarray([int(r[0]) for r in rows], dtype=np.int64)
    bid = np.asarray([float(r[1]) for r in rows], dtype=float)
    ask = np.asarray([float(r[2]) for r in rows], dtype=float)
    # feature_end is only used as an integrity assertion that samples and raw ticks overlap.
    if int(feature_start) < ts[0] or int(feature_end) > ts[-1] + DAY_NS:
        raise SystemExit("sample/tick day alignment failed")
    return ts, bid, ask


def _eligible_days(feature_ts, label_end_ts, *, min_day_samples: int,
                   min_train: int, calibration: int):
    days = []
    for day, val_start, val_end in _day_ranges(feature_ts):
        if val_end - val_start < min_day_samples:
            continue
        start_ts = int(feature_ts[val_start])
        pre = np.flatnonzero(label_end_ts[:val_start] < start_ts)
        if len(pre) >= min_train + calibration:
            days.append((day, val_start, val_end))
    return days


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only trade-level economic walk-forward for BBYG linear selective signals"
    )
    parser.add_argument("--min-train", type=int, default=10_000)
    parser.add_argument("--recent", type=int, default=20_000)
    parser.add_argument("--calibration", type=int, default=5_000)
    parser.add_argument("--min-day-samples", type=int, default=3_000)
    parser.add_argument("--linear-iterations", type=int, default=140)
    parser.add_argument("--seed", type=int, default=731022)
    args = parser.parse_args()
    if args.min_train < 2_000 or args.recent < args.min_train or args.calibration < 1_000:
        raise SystemExit("invalid training/calibration settings")

    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-multiday-v1"))
    store = ScalperStore(state_dir / "scalper.sqlite")
    try:
        sample_ids, feature_ts, label_end_ts, x, y = _load_samples(store)
        eligible = _eligible_days(
            feature_ts, label_end_ts,
            min_day_samples=args.min_day_samples,
            min_train=args.min_train,
            calibration=args.calibration,
        )
        if len(eligible) < 2:
            raise SystemExit("need at least two eligible out-of-sample days")
        latest_day = eligible[-1][0]

        policies = (
            EconomicPolicy(
                "portfolio_limits",
                max_positions=6,
                max_same_side_positions=4,
                max_entries_per_second=4,
                cooldown_ms=0,
            ),
            EconomicPolicy(
                "single_position",
                max_positions=1,
                max_same_side_positions=1,
                max_entries_per_second=4,
                cooldown_ms=0,
            ),
        )
        # Commission/slippage are intentionally expressed in entry-spread units because
        # broker-verified monetary commission is not yet available. The gross case isolates
        # strategy geometry; base/stress are sensitivity tests, not broker claims.
        scenarios = (
            CostScenario("gross", 0.0, 0.0),
            CostScenario("base_0p20", 0.05, 0.10),
            CostScenario("stress_0p40", 0.10, 0.20),
            CostScenario("severe_0p60", 0.20, 0.20),
        )
        label_settings = LabelSettings()

        day_outputs: list[dict] = []
        for fold_no, (day, val_start, val_end) in enumerate(eligible, start=1):
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

            eval_x = np.concatenate((x[cal_indices], x[val_start:val_end]), axis=0)
            p = fit_predict_architecture(
                "linear",
                x[train_indices],
                y[train_indices],
                eval_x,
                linear_iterations=args.linear_iterations,
                seed=args.seed + fold_no,
                restore_prior=False,
            )
            cal_p = p[:len(cal_indices)]
            val_p = p[len(cal_indices):]
            threshold, threshold_table = choose_selective_threshold(
                y[cal_indices], cal_p,
                min_coverage=0.05,
                min_selected=100,
                min_class_count=30,
            )
            if threshold is None:
                raise SystemExit(f"{day}: calibration could not select a threshold")

            tick_ts, bid, ask = _ticks_for_day(
                store,
                int(feature_ts[val_start]),
                int(feature_ts[val_end - 1]),
            )
            day_result = {
                "fold": fold_no,
                "utc_date": day,
                "is_latest_holdout": day == latest_day,
                "train_samples": int(len(train_indices)),
                "calibration_samples": int(len(cal_indices)),
                "validation_samples": int(val_end - val_start),
                "validation_start_sample_id": int(sample_ids[val_start]),
                "validation_end_sample_id": int(sample_ids[val_end - 1]),
                "chosen_threshold": float(threshold),
                "calibration_selected_candidate": next(
                    (row for row in threshold_table if row["threshold"] == threshold), None
                ),
                "policies": {},
            }
            for policy in policies:
                trades, stats = simulate_selective_trades(
                    tick_ts=tick_ts,
                    bid=bid,
                    ask=ask,
                    signal_ts=feature_ts[val_start:val_end],
                    probability_long=val_p,
                    threshold=float(threshold),
                    policy=policy,
                    label_settings=label_settings,
                )
                day_result["policies"][policy.name] = {
                    "flow": stats,
                    "cost_scenarios": {
                        scenario.name: economic_metrics(trades, scenario)
                        for scenario in scenarios
                    },
                }
            day_outputs.append(day_result)
            print(json.dumps({
                "stage": "day_complete",
                "utc_date": day,
                "threshold": float(threshold),
                "is_latest_holdout": day == latest_day,
                "portfolio_entries": day_result["policies"]["portfolio_limits"]["flow"]["entries_opened"],
                "single_entries": day_result["policies"]["single_position"]["flow"]["entries_opened"],
            }, sort_keys=True), flush=True)

        research_days = [d for d in day_outputs if not d["is_latest_holdout"]]
        holdout = next(d for d in day_outputs if d["is_latest_holdout"])
        summary = {}
        for policy in policies:
            summary[policy.name] = {}
            for scenario in scenarios:
                research_metrics = [
                    d["policies"][policy.name]["cost_scenarios"][scenario.name]
                    for d in research_days
                ]
                summary[policy.name][scenario.name] = aggregate_economic_metrics(research_metrics)

        print(json.dumps({
            "stage": "complete",
            "mode": "read_only_trade_level_economic_research",
            "execution_authorized": False,
            "commission_verified": False,
            "state_dir": str(state_dir),
            "samples": int(len(y)),
            "model": "linear_raw_balanced",
            "target_stop_contract": {
                "profit_spreads": label_settings.profit_spreads,
                "loss_spreads": label_settings.loss_spreads,
                "label_extra_cost_spreads": label_settings.extra_cost_spreads,
                "entry": "first_tick_strictly_after_signal",
                "target_fill": "barrier_price_no_favorable_overshoot_credit",
                "stop_fill": "executable_quote_with_adverse_gap_preserved",
            },
            "settings": {
                "min_train": args.min_train,
                "recent": args.recent,
                "calibration": args.calibration,
                "min_day_samples": args.min_day_samples,
                "linear_iterations": args.linear_iterations,
            },
            "cost_scenarios": {
                s.name: {
                    "slippage_per_side_spreads": s.slippage_per_side_spreads,
                    "commission_roundturn_spreads": s.commission_roundturn_spreads,
                    "total_extra_cost_spreads": s.total_extra_cost_spreads,
                } for s in scenarios
            },
            "research_dates": [d["utc_date"] for d in research_days],
            "latest_holdout_date": holdout["utc_date"],
            "summary": summary,
            "latest_holdout": holdout,
            "days": day_outputs,
        }, sort_keys=True), flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
