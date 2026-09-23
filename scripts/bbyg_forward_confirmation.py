from __future__ import annotations

from datetime import datetime, timezone
import argparse
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
FREEZE_DATE = "2026-09-22"
THRESHOLDS = (0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60)


def _load_samples(store: ScalperStore):
    count = store.sample_count()
    interval_count = int(store.db.execute("SELECT count(*) FROM sample_label_intervals").fetchone()[0])
    if count == 0 or interval_count != count:
        raise SystemExit(f"incomplete sample dataset: samples={count}, intervals={interval_count}")
    feature_ts = np.empty(count, dtype=np.int64)
    label_end_ts = np.empty(count, dtype=np.int64)
    x = np.empty((count, 8), dtype=float)
    y = np.empty(count, dtype=np.int8)
    loaded = 0
    for loaded, (ts_ns, x_json, label, end_ts) in enumerate(store.db.execute(
        """SELECT s.feature_ts_ns,s.x_json,s.y,i.label_end_ts_ns
           FROM samples s JOIN sample_label_intervals i
           ON i.feature_ts_ns=s.feature_ts_ns ORDER BY s.id"""
    ), start=1):
        pos = loaded - 1
        values = json.loads(x_json)
        if len(values) != 8:
            raise SystemExit("unexpected feature width")
        feature_ts[pos] = int(ts_ns)
        label_end_ts[pos] = int(end_ts)
        x[pos] = np.asarray(values, dtype=float)
        y[pos] = int(label)
    if loaded != count or not np.isfinite(x).all():
        raise SystemExit("sample load failed")
    return feature_ts, label_end_ts, x, y


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


def _ticks_for_day(store: ScalperStore, ts_ns: int):
    start = (int(ts_ns) // DAY_NS) * DAY_NS
    return _ticks_between(store, start, start + DAY_NS - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen BBYG forward confirmation evaluator")
    parser.add_argument("--include-latest-date", action="store_true",
                        help="score latest stored UTC date even if it may be partial")
    args = parser.parse_args()

    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-multiday-v1"))
    store = ScalperStore(state_dir / "scalper.sqlite")
    try:
        feature_ts, label_end_ts, x, y = _load_samples(store)
        ranges = _day_ranges(feature_ts)
        latest_stored_date = ranges[-1][0]
        future_ranges = [r for r in ranges if r[0] > FREEZE_DATE]
        if not args.include_latest_date:
            future_ranges = [r for r in future_ranges if r[0] != latest_stored_date]

        settings = LabelSettings(
            profit_spreads=1.6,
            loss_spreads=1.4,
            extra_cost_spreads=0.20,
            max_lookahead_ticks=600,
            max_entry_delay_seconds=5.0,
            stop_reference="entry",
        )
        policy = EconomicPolicy(
            "portfolio_limits",
            max_positions=6,
            max_same_side_positions=4,
            max_entries_per_second=4,
            cooldown_ms=0,
            max_entry_delay_seconds=5.0,
            max_gap_seconds=300.0,
        )
        scenario = CostScenario("base_0p20", 0.05, 0.10)

        outputs = []
        for fold_no, (day, val_start, val_end) in enumerate(future_ranges, start=1):
            if val_end - val_start < 3000:
                outputs.append({"utc_date": day, "status": "insufficient_day_samples",
                                "validation_samples": int(val_end - val_start)})
                continue
            validation_start_ts = int(feature_ts[val_start])
            pre = np.flatnonzero(label_end_ts[:val_start] < validation_start_ts)
            if len(pre) < 15_000:
                outputs.append({"utc_date": day, "status": "insufficient_history"})
                continue
            cal_indices = pre[-5000:]
            calibration_start_index = int(cal_indices[0])
            calibration_start_ts = int(feature_ts[calibration_start_index])
            train_indices = np.flatnonzero(label_end_ts[:calibration_start_index] < calibration_start_ts)
            if len(train_indices) > 20_000:
                train_indices = train_indices[-20_000:]
            if len(train_indices) < 10_000:
                outputs.append({"utc_date": day, "status": "insufficient_train"})
                continue

            eval_x = np.concatenate((x[cal_indices], x[val_start:val_end]), axis=0)
            p = fit_predict_architecture(
                "linear", x[train_indices], y[train_indices], eval_x,
                linear_iterations=140,
                seed=731022 + fold_no,
                restore_prior=False,
            )
            cal_p = p[:len(cal_indices)]
            val_p = p[len(cal_indices):]

            cal_tick_ts, cal_bid, cal_ask = _ticks_between(
                store,
                int(feature_ts[cal_indices[0]]) - 5_000_000_000,
                int(feature_ts[cal_indices[-1]]) + 300_000_000_000,
            )
            candidates = []
            calibration = []
            for threshold in THRESHOLDS:
                trades, _flow = simulate_selective_trades(
                    tick_ts=cal_tick_ts,
                    bid=cal_bid,
                    ask=cal_ask,
                    signal_ts=feature_ts[cal_indices],
                    probability_long=cal_p,
                    threshold=threshold,
                    policy=policy,
                    label_settings=settings,
                )
                metrics = economic_metrics(trades, scenario)
                candidate = EconomicThresholdCandidate(
                    threshold=float(threshold),
                    trades=int(metrics["trades"]),
                    net_pnl_spreads=float(metrics["net_pnl_spreads"]),
                    average_net_pnl_spreads=(None if metrics["average_net_pnl_spreads"] is None
                                             else float(metrics["average_net_pnl_spreads"])),
                    profit_factor=(None if metrics["profit_factor"] is None
                                   else float(metrics["profit_factor"])),
                    win_rate=(None if metrics["win_rate"] is None else float(metrics["win_rate"])),
                )
                candidates.append(candidate)
                calibration.append({
                    "threshold": threshold,
                    "trades": candidate.trades,
                    "net_pnl_spreads": candidate.net_pnl_spreads,
                    "profit_factor": candidate.profit_factor,
                })

            chosen = choose_economic_threshold(candidates, min_trades=100, min_profit_factor=1.05)
            result = {
                "utc_date": day,
                "status": "abstained" if chosen is None else "traded",
                "train_samples": int(len(train_indices)),
                "calibration_samples": int(len(cal_indices)),
                "validation_samples": int(val_end - val_start),
                "chosen_threshold": None if chosen is None else chosen.threshold,
                "calibration": calibration,
                "validation": None,
            }
            if chosen is not None:
                tick_ts, bid, ask = _ticks_for_day(store, int(feature_ts[val_start]))
                trades, flow = simulate_selective_trades(
                    tick_ts=tick_ts,
                    bid=bid,
                    ask=ask,
                    signal_ts=feature_ts[val_start:val_end],
                    probability_long=val_p,
                    threshold=chosen.threshold,
                    policy=policy,
                    label_settings=settings,
                )
                result["validation"] = {
                    "flow": flow,
                    "base_0p20": economic_metrics(trades, scenario),
                    "gross": economic_metrics(trades, CostScenario("gross", 0.0, 0.0)),
                }
            outputs.append(result)
            print(json.dumps({
                "stage": "forward_day_complete",
                "utc_date": day,
                "status": result["status"],
                "chosen_threshold": result["chosen_threshold"],
                "validation_trades": 0 if result["validation"] is None else result["validation"]["base_0p20"]["trades"],
            }, sort_keys=True), flush=True)

        traded = [r for r in outputs if r.get("status") == "traded" and r.get("validation") is not None]
        metrics = [r["validation"]["base_0p20"] for r in traded]
        print(json.dumps({
            "stage": "complete",
            "mode": "frozen_forward_confirmation_v1",
            "execution_authorized": False,
            "freeze_date": FREEZE_DATE,
            "latest_stored_date": latest_stored_date,
            "latest_date_excluded": not args.include_latest_date,
            "confirmation_days_considered": len(outputs),
            "traded_days": len(traded),
            "abstained_days": sum(r.get("status") == "abstained" for r in outputs),
            "positive_traded_days": sum(float(m["net_pnl_spreads"]) > 0 for m in metrics),
            "total_trades": sum(int(m["trades"]) for m in metrics),
            "total_net_pnl_spreads": sum(float(m["net_pnl_spreads"]) for m in metrics),
            "days": outputs,
        }, sort_keys=True), flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
