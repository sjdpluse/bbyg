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


def _day_start_ns(day: str) -> int:
    dt = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def _ticks_between_arrays(ts, bid, ask, start_ns: int, end_ns: int):
    left = int(np.searchsorted(ts, int(start_ns), side="left"))
    right = int(np.searchsorted(ts, int(end_ns), side="right"))
    if right - left < 2:
        raise SystemExit("insufficient raw ticks for parity replay window")
    return ts[left:right], bid[left:right], ask[left:right]


def _candidate(threshold: float, metrics: dict) -> EconomicThresholdCandidate:
    return EconomicThresholdCandidate(
        threshold=float(threshold),
        trades=int(metrics["trades"]),
        net_pnl_spreads=float(metrics["net_pnl_spreads"]),
        average_net_pnl_spreads=(None if metrics["average_net_pnl_spreads"] is None
                                 else float(metrics["average_net_pnl_spreads"])),
        profit_factor=(None if metrics["profit_factor"] is None else float(metrics["profit_factor"])),
        win_rate=(None if metrics["win_rate"] is None else float(metrics["win_rate"])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BBYG v1 persisted-feature parity against v2 all-anchor features")
    parser.add_argument("--day", default="2026-09-23", help="already-inspected UTC day used only for parity diagnostics")
    args = parser.parse_args()

    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-multiday-v1"))
    store = ScalperStore(state_dir / "scalper.sqlite")
    try:
        feature_ts, label_end_ts, persisted_x, y = _load_samples(store)
        replay = FastGapAwareReplayBuilder(stride=4, max_gap_seconds=300.0)
        tick_ts, bid, ask = replay._load_arrays(store)
        anchors, gap_starts = replay._anchors(tick_ts)
        anchor_ts = tick_ts[anchors]
        anchor_x = replay._feature_matrix(anchors, bid, ask)

        # Every persisted labeled sample should be a causal anchor. Rebuild its feature vector
        # from raw ticks without using any future label information.
        positions = np.searchsorted(anchor_ts, feature_ts)
        in_bounds = positions < len(anchor_ts)
        matched = np.zeros(len(feature_ts), dtype=bool)
        matched[in_bounds] = anchor_ts[positions[in_bounds]] == feature_ts[in_bounds]
        matched_count = int(matched.sum())
        if matched_count:
            delta = np.abs(persisted_x[matched] - anchor_x[positions[matched]])
            max_abs = float(delta.max())
            mean_abs = float(delta.mean())
            p999_abs = float(np.quantile(delta, 0.999))
        else:
            max_abs = mean_abs = p999_abs = float("inf")

        feature_parity = {
            "persisted_samples": int(len(feature_ts)),
            "matched_anchor_timestamps": matched_count,
            "unmatched_anchor_timestamps": int(len(feature_ts) - matched_count),
            "max_abs_feature_error": max_abs,
            "mean_abs_feature_error": mean_abs,
            "p999_abs_feature_error": p999_abs,
            "pass": bool(matched_count == len(feature_ts) and max_abs <= 1e-12),
        }
        print(json.dumps({"stage": "feature_parity", **feature_parity}, sort_keys=True), flush=True)

        day_start = _day_start_ns(args.day)
        pre = np.flatnonzero(label_end_ts < day_start)
        if len(pre) < 15_000:
            raise SystemExit("insufficient pre-day labeled history for parity audit")
        cal_indices = pre[-5000:]
        cal_start_index = int(cal_indices[0])
        cal_start_ts = int(feature_ts[cal_start_index])
        train_indices = np.flatnonzero(label_end_ts[:cal_start_index] < cal_start_ts)
        if len(train_indices) > 20_000:
            train_indices = train_indices[-20_000:]
        if len(train_indices) < 10_000:
            raise SystemExit("insufficient training history for parity audit")
        if not np.all(matched[cal_indices]):
            raise SystemExit("calibration labeled timestamps are missing from all-anchor universe")

        recomputed_cal_x = anchor_x[positions[cal_indices]]
        persisted_p = fit_predict_architecture(
            "linear", persisted_x[train_indices], y[train_indices], persisted_x[cal_indices],
            linear_iterations=140, seed=731023, restore_prior=False,
        )
        recomputed_p = fit_predict_architecture(
            "linear", persisted_x[train_indices], y[train_indices], recomputed_cal_x,
            linear_iterations=140, seed=731023, restore_prior=False,
        )
        p_delta = np.abs(persisted_p - recomputed_p)
        probability_parity = {
            "calibration_samples": int(len(cal_indices)),
            "max_abs_probability_error": float(p_delta.max()),
            "mean_abs_probability_error": float(p_delta.mean()),
            "pass": bool(float(p_delta.max()) <= 1e-12),
        }
        print(json.dumps({"stage": "probability_parity", **probability_parity}, sort_keys=True), flush=True)

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
        cal_tick_ts, cal_bid, cal_ask = _ticks_between_arrays(
            tick_ts, bid, ask,
            int(feature_ts[cal_indices[0]]) - 5_000_000_000,
            int(feature_ts[cal_indices[-1]]) + 300_000_000_000,
        )

        persisted_candidates = []
        recomputed_candidates = []
        rows = []
        replay_exact = True
        for threshold in THRESHOLDS:
            trades_a, flow_a = simulate_selective_trades(
                tick_ts=cal_tick_ts, bid=cal_bid, ask=cal_ask,
                signal_ts=feature_ts[cal_indices], probability_long=persisted_p,
                threshold=threshold, policy=policy, label_settings=settings,
            )
            trades_b, flow_b = simulate_selective_trades(
                tick_ts=cal_tick_ts, bid=cal_bid, ask=cal_ask,
                signal_ts=feature_ts[cal_indices], probability_long=recomputed_p,
                threshold=threshold, policy=policy, label_settings=settings,
            )
            ma = economic_metrics(trades_a, scenario)
            mb = economic_metrics(trades_b, scenario)
            ca = _candidate(threshold, ma)
            cb = _candidate(threshold, mb)
            persisted_candidates.append(ca)
            recomputed_candidates.append(cb)
            pf_a = ca.profit_factor
            pf_b = cb.profit_factor
            row_equal = (
                ca.trades == cb.trades
                and abs(ca.net_pnl_spreads - cb.net_pnl_spreads) <= 1e-9
                and ((pf_a is None and pf_b is None)
                     or (pf_a is not None and pf_b is not None and abs(pf_a - pf_b) <= 1e-12))
                and int(flow_a["signals_selected"]) == int(flow_b["signals_selected"])
            )
            replay_exact &= row_equal
            rows.append({
                "threshold": threshold,
                "persisted_trades": ca.trades,
                "recomputed_trades": cb.trades,
                "persisted_net_pnl_spreads": ca.net_pnl_spreads,
                "recomputed_net_pnl_spreads": cb.net_pnl_spreads,
                "persisted_profit_factor": pf_a,
                "recomputed_profit_factor": pf_b,
                "persisted_signals_selected": int(flow_a["signals_selected"]),
                "recomputed_signals_selected": int(flow_b["signals_selected"]),
                "exact": bool(row_equal),
            })

        chosen_a = choose_economic_threshold(persisted_candidates, min_trades=100, min_profit_factor=1.05)
        chosen_b = choose_economic_threshold(recomputed_candidates, min_trades=100, min_profit_factor=1.05)
        chosen_a_value = None if chosen_a is None else float(chosen_a.threshold)
        chosen_b_value = None if chosen_b is None else float(chosen_b.threshold)
        calibration_parity = {
            "day": args.day,
            "train_samples": int(len(train_indices)),
            "calibration_samples": int(len(cal_indices)),
            "persisted_chosen_threshold": chosen_a_value,
            "recomputed_chosen_threshold": chosen_b_value,
            "rows": rows,
            "pass": bool(replay_exact and chosen_a_value == chosen_b_value),
        }
        print(json.dumps({"stage": "calibration_parity", **calibration_parity}, sort_keys=True), flush=True)

        overall = bool(feature_parity["pass"] and probability_parity["pass"] and calibration_parity["pass"])
        print(json.dumps({
            "stage": "complete",
            "audit": "bbyg_v1_v2_feature_and_labeled_universe_parity",
            "utc_day": args.day,
            "market_gaps": int(len(gap_starts)),
            "pass": overall,
            "interpretation": (
                "If pass=true, v2 feature/model mechanics reproduce v1 exactly on the same labeled signal universe; "
                "the large v1-v2 difference is therefore attributable to expanding the signal universe to all causal anchors."
                if overall else
                "Parity failed. Do not interpret v1-v2 performance differences until the mismatch is located."
            ),
            "execution_authorized": False,
        }, sort_keys=True), flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
