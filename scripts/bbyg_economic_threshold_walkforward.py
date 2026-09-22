from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from truetrade.scalper.economic_gate import EconomicThresholdCandidate, choose_economic_threshold
from truetrade.scalper.economic_replay import CostScenario, EconomicPolicy, economic_metrics, simulate_selective_trades
from truetrade.scalper.labels import LabelSettings
from truetrade.scalper.research_models import fit_predict_architecture
from truetrade.scalper.store import ScalperStore

DAY_NS = 86_400_000_000_000
THRESHOLDS = (0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60)


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
        pos = loaded - 1
        values = json.loads(x_json)
        if len(values) != 8:
            raise SystemExit(f"unexpected feature width at sample {sid}: {len(values)}")
        sample_ids[pos] = int(sid)
        feature_ts[pos] = int(ts_ns)
        label_end_ts[pos] = int(end_ts)
        x[pos] = np.asarray(values, dtype=float)
        y[pos] = int(label)
    if loaded != count or not np.isfinite(x).all():
        raise SystemExit("sample load failed")
    return sample_ids, feature_ts, label_end_ts, x, y


def _day_ranges(feature_ts: np.ndarray):
    keys = feature_ts // DAY_NS
    boundaries = np.concatenate((
        np.asarray([0], dtype=np.int64),
        np.flatnonzero(np.diff(keys) != 0).astype(np.int64) + 1,
        np.asarray([len(feature_ts)], dtype=np.int64),
    ))
    out = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        left, right = int(a), int(b)
        day = datetime.fromtimestamp(feature_ts[left] / 1_000_000_000, tz=timezone.utc).date().isoformat()
        out.append((day, left, right))
    return out


def _ticks_between(store: ScalperStore, start_ns: int, end_ns: int):
    rows = list(store.db.execute(
        "SELECT ts_ns,bid,ask FROM ticks WHERE ts_ns>=? AND ts_ns<=? ORDER BY ts_ns",
        (int(start_ns), int(end_ns)),
    ))
    if len(rows) < 2:
        raise SystemExit("insufficient raw ticks for replay window")
    return (
        np.asarray([int(r[0]) for r in rows], dtype=np.int64),
        np.asarray([float(r[1]) for r in rows], dtype=float),
        np.asarray([float(r[2]) for r in rows], dtype=float),
    )


def _ticks_for_utc_day(store: ScalperStore, sample_ts_ns: int):
    day_start = (int(sample_ts_ns) // DAY_NS) * DAY_NS
    day_end = day_start + DAY_NS - 1
    return _ticks_between(store, day_start, day_end)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-safe economic-threshold walk-forward for BBYG linear signals"
    )
    parser.add_argument("--min-train", type=int, default=10_000)
    parser.add_argument("--recent", type=int, default=20_000)
    parser.add_argument("--calibration", type=int, default=5_000)
    parser.add_argument("--min-day-samples", type=int, default=3_000)
    parser.add_argument("--linear-iterations", type=int, default=140)
    parser.add_argument("--min-calibration-trades", type=int, default=100)
    parser.add_argument("--min-calibration-profit-factor", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=731022)
    args = parser.parse_args()

    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-multiday-v1"))
    store = ScalperStore(state_dir / "scalper.sqlite")
    try:
        sample_ids, feature_ts, label_end_ts, x, y = _load_samples(store)
        eligible = []
        for day, val_start, val_end in _day_ranges(feature_ts):
            if val_end - val_start < args.min_day_samples:
                continue
            validation_start_ts = int(feature_ts[val_start])
            pre = np.flatnonzero(label_end_ts[:val_start] < validation_start_ts)
            if len(pre) >= args.min_train + args.calibration:
                eligible.append((day, val_start, val_end))
        if len(eligible) < 2:
            raise SystemExit("need at least two eligible out-of-sample days")

        latest_day = eligible[-1][0]
        policy = EconomicPolicy(
            "portfolio_limits", max_positions=6, max_same_side_positions=4,
            max_entries_per_second=4, cooldown_ms=0,
        )
        scenario = CostScenario("base_0p20", 0.05, 0.10)
        label_settings = LabelSettings()
        outputs = []

        for fold_no, (day, val_start, val_end) in enumerate(eligible, start=1):
            validation_start_ts = int(feature_ts[val_start])
            pre = np.flatnonzero(label_end_ts[:val_start] < validation_start_ts)
            cal_indices = pre[-args.calibration:]
            calibration_start_index = int(cal_indices[0])
            calibration_start_ts = int(feature_ts[calibration_start_index])
            train_indices = np.flatnonzero(label_end_ts[:calibration_start_index] < calibration_start_ts)
            if len(train_indices) > args.recent:
                train_indices = train_indices[-args.recent:]
            if len(train_indices) < args.min_train:
                continue

            eval_x = np.concatenate((x[cal_indices], x[val_start:val_end]), axis=0)
            p = fit_predict_architecture(
                "linear", x[train_indices], y[train_indices], eval_x,
                linear_iterations=args.linear_iterations,
                seed=args.seed + fold_no,
                restore_prior=False,
            )
            cal_p = p[:len(cal_indices)]
            val_p = p[len(cal_indices):]

            # Replay calibration on its exact historical tick interval. Signals are already
            # leakage-safe because cal_indices were chosen only from labels resolved before
            # the validation day, and model training ends before calibration starts.
            cal_tick_ts, cal_bid, cal_ask = _ticks_between(
                store,
                int(feature_ts[cal_indices[0]]) - 5_000_000_000,
                int(feature_ts[cal_indices[-1]]) + 300_000_000_000,
            )
            candidates = []
            calibration_table = []
            for threshold in THRESHOLDS:
                cal_trades, cal_flow = simulate_selective_trades(
                    tick_ts=cal_tick_ts,
                    bid=cal_bid,
                    ask=cal_ask,
                    signal_ts=feature_ts[cal_indices],
                    probability_long=cal_p,
                    threshold=threshold,
                    policy=policy,
                    label_settings=label_settings,
                )
                metrics = economic_metrics(cal_trades, scenario)
                candidate = EconomicThresholdCandidate(
                    threshold=float(threshold),
                    trades=int(metrics["trades"]),
                    net_pnl_spreads=float(metrics["net_pnl_spreads"]),
                    average_net_pnl_spreads=(
                        None if metrics["average_net_pnl_spreads"] is None
                        else float(metrics["average_net_pnl_spreads"])
                    ),
                    profit_factor=(
                        None if metrics["profit_factor"] is None else float(metrics["profit_factor"])
                    ),
                    win_rate=None if metrics["win_rate"] is None else float(metrics["win_rate"]),
                )
                candidates.append(candidate)
                calibration_table.append({
                    "threshold": threshold,
                    "flow": cal_flow,
                    "metrics": metrics,
                    "economically_positive": candidate.economically_positive,
                })

            chosen = choose_economic_threshold(
                candidates,
                min_trades=args.min_calibration_trades,
                min_profit_factor=args.min_calibration_profit_factor,
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
                "calibration_table": calibration_table,
                "chosen_threshold": None if chosen is None else chosen.threshold,
                "validation": None,
            }

            if chosen is not None:
                tick_ts, bid, ask = _ticks_for_utc_day(store, int(feature_ts[val_start]))
                trades, flow = simulate_selective_trades(
                    tick_ts=tick_ts,
                    bid=bid,
                    ask=ask,
                    signal_ts=feature_ts[val_start:val_end],
                    probability_long=val_p,
                    threshold=chosen.threshold,
                    policy=policy,
                    label_settings=label_settings,
                )
                day_result["validation"] = {
                    "flow": flow,
                    "base_0p20": economic_metrics(trades, scenario),
                    "gross": economic_metrics(trades, CostScenario("gross", 0.0, 0.0)),
                }
            outputs.append(day_result)
            print(json.dumps({
                "stage": "day_complete",
                "utc_date": day,
                "is_latest_holdout": day == latest_day,
                "chosen_threshold": None if chosen is None else chosen.threshold,
                "validation_trades": (
                    0 if day_result["validation"] is None
                    else day_result["validation"]["base_0p20"]["trades"]
                ),
            }, sort_keys=True), flush=True)

        research = [d for d in outputs if not d["is_latest_holdout"]]
        holdout = next(d for d in outputs if d["is_latest_holdout"])
        active_research = [d for d in research if d["validation"] is not None]
        summary = {
            "research_days": len(research),
            "research_days_with_economic_threshold": len(active_research),
            "research_days_abstained": len(research) - len(active_research),
            "positive_validation_days": int(sum(
                d["validation"] is not None
                and float(d["validation"]["base_0p20"]["net_pnl_spreads"]) > 0
                for d in research
            )),
            "total_validation_trades": int(sum(
                0 if d["validation"] is None else int(d["validation"]["base_0p20"]["trades"])
                for d in research
            )),
            "total_validation_net_pnl_spreads": float(sum(
                0.0 if d["validation"] is None else float(d["validation"]["base_0p20"]["net_pnl_spreads"])
                for d in research
            )),
            "holdout_traded": holdout["validation"] is not None,
            "holdout_base_0p20": None if holdout["validation"] is None else holdout["validation"]["base_0p20"],
        }

        print(json.dumps({
            "stage": "complete",
            "mode": "read_only_economic_threshold_research",
            "execution_authorized": False,
            "commission_verified": False,
            "state_dir": str(state_dir),
            "model": "linear_raw_balanced",
            "cost_scenario": {
                "name": scenario.name,
                "total_extra_cost_spreads": scenario.total_extra_cost_spreads,
            },
            "thresholds": list(THRESHOLDS),
            "settings": {
                "min_train": args.min_train,
                "recent": args.recent,
                "calibration": args.calibration,
                "min_calibration_trades": args.min_calibration_trades,
                "min_calibration_profit_factor": args.min_calibration_profit_factor,
            },
            "summary": summary,
            "latest_holdout": holdout,
            "days": outputs,
        }, sort_keys=True), flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
