from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path

import numpy as np

from truetrade.scalper.economic_gate import EconomicThresholdCandidate, choose_economic_threshold
from truetrade.scalper.economic_replay import CostScenario, EconomicPolicy, economic_metrics, simulate_selective_trades
from truetrade.scalper.fast_replay import FastGapAwareReplayBuilder
from truetrade.scalper.labels import LabelSettings
from truetrade.scalper.research_models import fit_predict_architecture
from truetrade.scalper.store import ScalperStore

DAY_NS = 86_400_000_000_000
V1_FREEZE_DATE = "2026-09-22"
V2_PRISTINE_AFTER_DATE = "2026-09-23"
THRESHOLDS = (0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60)


def _load_labeled_samples(store: ScalperStore):
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


def _date_string(ts_ns: int) -> str:
    return datetime.fromtimestamp(int(ts_ns) / 1_000_000_000, tz=timezone.utc).date().isoformat()


def _ticks_for_day(ts: np.ndarray, bid: np.ndarray, ask: np.ndarray, day_start_ns: int):
    left = int(np.searchsorted(ts, day_start_ns, side="left"))
    right = int(np.searchsorted(ts, day_start_ns + DAY_NS, side="left"))
    if right - left < 2:
        raise SystemExit("insufficient raw ticks for replay day")
    return ts[left:right], bid[left:right], ask[left:right]


def main() -> None:
    parser = argparse.ArgumentParser(description="BBYG all-anchor forward confirmation evaluator v2")
    parser.add_argument("--include-latest-date", action="store_true",
                        help="score latest stored UTC date even if it may be partial")
    args = parser.parse_args()

    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-multiday-v1"))
    store = ScalperStore(state_dir / "scalper.sqlite")
    try:
        labeled_ts, label_end_ts, labeled_x, y = _load_labeled_samples(store)

        replay = FastGapAwareReplayBuilder(stride=4, max_gap_seconds=300.0)
        tick_ts, bid, ask = replay._load_arrays(store)
        anchors, gap_starts = replay._anchors(tick_ts)
        anchor_x = replay._feature_matrix(anchors, bid, ask)
        anchor_ts = tick_ts[anchors]
        if len(anchor_ts) == 0:
            raise SystemExit("no causal anchors available")

        latest_stored_date = _date_string(int(tick_ts[-1]))
        unique_days = np.unique(anchor_ts // DAY_NS)
        future_days = [int(k) for k in unique_days if _date_string(int(k) * DAY_NS) > V1_FREEZE_DATE]
        if not args.include_latest_date:
            future_days = [k for k in future_days if _date_string(k * DAY_NS) != latest_stored_date]

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
            max_holding_ticks=600,
        )
        scenario = CostScenario("base_0p20", 0.05, 0.10)

        outputs = []
        for fold_no, day_key in enumerate(future_days, start=1):
            day_start = int(day_key * DAY_NS)
            day_end = day_start + DAY_NS
            day = _date_string(day_start)
            val_mask = (anchor_ts >= day_start) & (anchor_ts < day_end)
            val_anchor_idx = np.flatnonzero(val_mask)
            if len(val_anchor_idx) < 3000:
                outputs.append({"utc_date": day, "status": "insufficient_day_anchors",
                                "validation_anchors": int(len(val_anchor_idx))})
                continue

            validation_start_ts = day_start
            pre = np.flatnonzero(label_end_ts < validation_start_ts)
            if len(pre) < 15_000:
                outputs.append({"utc_date": day, "status": "insufficient_history"})
                continue
            cal_labeled_indices = pre[-5000:]
            cal_start_ts = int(labeled_ts[cal_labeled_indices[0]])
            cal_last_labeled_ts = int(labeled_ts[cal_labeled_indices[-1]])

            train_indices = np.flatnonzero(label_end_ts < cal_start_ts)
            if len(train_indices) > 20_000:
                train_indices = train_indices[-20_000:]
            if len(train_indices) < 10_000:
                outputs.append({"utc_date": day, "status": "insufficient_train"})
                continue

            # All eligible causal stride anchors in the calibration time window are signals.
            # Conservatively require the full 600-tick evidence horizon to end before validation.
            cal_anchor_mask = (anchor_ts >= cal_start_ts) & (anchor_ts <= cal_last_labeled_ts)
            cal_anchor_idx = np.flatnonzero(cal_anchor_mask)
            full_horizon = anchors[cal_anchor_idx] + 1 + settings.max_lookahead_ticks < len(tick_ts)
            cal_anchor_idx = cal_anchor_idx[full_horizon]
            if len(cal_anchor_idx):
                horizon_end_idx = anchors[cal_anchor_idx] + 1 + settings.max_lookahead_ticks
                cal_anchor_idx = cal_anchor_idx[tick_ts[horizon_end_idx] < validation_start_ts]
            if len(cal_anchor_idx) < 100:
                outputs.append({"utc_date": day, "status": "insufficient_calibration_anchors",
                                "calibration_anchors": int(len(cal_anchor_idx))})
                continue

            eval_x = np.concatenate((anchor_x[cal_anchor_idx], anchor_x[val_anchor_idx]), axis=0)
            p = fit_predict_architecture(
                "linear", labeled_x[train_indices], y[train_indices], eval_x,
                linear_iterations=140,
                seed=731022 + fold_no,
                restore_prior=False,
            )
            cal_p = p[:len(cal_anchor_idx)]
            val_p = p[len(cal_anchor_idx):]

            cal_tick_start = max(0, int(anchors[cal_anchor_idx[0]]) - 2)
            cal_tick_end = min(len(tick_ts), int(anchors[cal_anchor_idx[-1]]) + 1 + settings.max_lookahead_ticks + 2)
            cal_tick_ts = tick_ts[cal_tick_start:cal_tick_end]
            cal_bid = bid[cal_tick_start:cal_tick_end]
            cal_ask = ask[cal_tick_start:cal_tick_end]

            candidates = []
            calibration = []
            for threshold in THRESHOLDS:
                trades, flow = simulate_selective_trades(
                    tick_ts=cal_tick_ts,
                    bid=cal_bid,
                    ask=cal_ask,
                    signal_ts=anchor_ts[cal_anchor_idx],
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
                    "signals_total": int(flow["signals_total"]),
                    "signals_selected": int(flow["signals_selected"]),
                    "trades": candidate.trades,
                    "net_pnl_spreads": candidate.net_pnl_spreads,
                    "profit_factor": candidate.profit_factor,
                })

            chosen = choose_economic_threshold(candidates, min_trades=100, min_profit_factor=1.05)
            result = {
                "utc_date": day,
                "evidence_class": "pristine_v2" if day > V2_PRISTINE_AFTER_DATE else "diagnostic_v2_crosscheck",
                "status": "abstained" if chosen is None else "traded",
                "train_labeled_samples": int(len(train_indices)),
                "calibration_reference_labeled_samples": int(len(cal_labeled_indices)),
                "calibration_all_anchors": int(len(cal_anchor_idx)),
                "validation_all_anchors": int(len(val_anchor_idx)),
                "chosen_threshold": None if chosen is None else chosen.threshold,
                "calibration": calibration,
                "validation": None,
            }
            if chosen is not None:
                day_tick_ts, day_bid, day_ask = _ticks_for_day(tick_ts, bid, ask, day_start)
                trades, flow = simulate_selective_trades(
                    tick_ts=day_tick_ts,
                    bid=day_bid,
                    ask=day_ask,
                    signal_ts=anchor_ts[val_anchor_idx],
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
                "stage": "forward_v2_day_complete",
                "utc_date": day,
                "evidence_class": result["evidence_class"],
                "status": result["status"],
                "chosen_threshold": result["chosen_threshold"],
                "validation_anchors": result["validation_all_anchors"],
                "validation_trades": 0 if result["validation"] is None else result["validation"]["base_0p20"]["trades"],
            }, sort_keys=True), flush=True)

        traded = [r for r in outputs if r.get("status") == "traded" and r.get("validation") is not None]
        metrics = [r["validation"]["base_0p20"] for r in traded]
        print(json.dumps({
            "stage": "complete",
            "mode": "bbyg_forward_confirmation_v2_all_anchor",
            "execution_authorized": False,
            "v1_freeze_date": V1_FREEZE_DATE,
            "v2_pristine_after_utc_date": V2_PRISTINE_AFTER_DATE,
            "latest_stored_date": latest_stored_date,
            "latest_date_excluded": not args.include_latest_date,
            "signal_universe": "all_causal_stride_anchors",
            "total_causal_anchors": int(len(anchors)),
            "market_gaps": int(len(gap_starts)),
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
