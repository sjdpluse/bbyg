from __future__ import annotations

from dataclasses import dataclass
from collections import deque

import numpy as np

from .labels import LabelSettings


@dataclass(frozen=True)
class EconomicPolicy:
    name: str
    max_positions: int = 6
    max_same_side_positions: int = 4
    max_entries_per_second: int = 4
    cooldown_ms: int = 250
    max_entry_delay_seconds: float = 5.0
    max_gap_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.max_positions < 1 or self.max_same_side_positions < 1:
            raise ValueError("position limits must be positive")
        if self.max_same_side_positions > self.max_positions:
            raise ValueError("same-side limit cannot exceed total limit")
        if self.max_entries_per_second < 1:
            raise ValueError("max_entries_per_second must be positive")
        if self.cooldown_ms < 0 or self.max_entry_delay_seconds <= 0 or self.max_gap_seconds <= 0:
            raise ValueError("invalid timing limits")


@dataclass(frozen=True)
class CostScenario:
    name: str
    slippage_per_side_spreads: float
    commission_roundturn_spreads: float

    def __post_init__(self) -> None:
        if self.slippage_per_side_spreads < 0 or self.commission_roundturn_spreads < 0:
            raise ValueError("cost assumptions cannot be negative")

    @property
    def total_extra_cost_spreads(self) -> float:
        return 2.0 * self.slippage_per_side_spreads + self.commission_roundturn_spreads


@dataclass
class OpenTrade:
    trade_id: int
    signal_ts_ns: int
    entry_index: int
    entry_ts_ns: int
    side: int
    confidence: float
    entry: float
    entry_spread: float
    target: float
    stop: float


@dataclass(frozen=True)
class ClosedTrade:
    trade_id: int
    signal_ts_ns: int
    entry_ts_ns: int
    exit_ts_ns: int
    side: int
    confidence: float
    entry: float
    exit: float
    entry_spread: float
    gross_pnl_spreads: float
    hold_ticks: int
    hold_ms: float
    exit_reason: str


def _close_trade(position: OpenTrade, *, exit_price: float, exit_index: int,
                 exit_ts_ns: int, reason: str) -> ClosedTrade:
    pnl = (float(exit_price) - position.entry) * position.side
    gross = pnl / max(position.entry_spread, 1e-12)
    return ClosedTrade(
        trade_id=position.trade_id,
        signal_ts_ns=position.signal_ts_ns,
        entry_ts_ns=position.entry_ts_ns,
        exit_ts_ns=int(exit_ts_ns),
        side=position.side,
        confidence=position.confidence,
        entry=position.entry,
        exit=float(exit_price),
        entry_spread=position.entry_spread,
        gross_pnl_spreads=float(gross),
        hold_ticks=int(exit_index - position.entry_index),
        hold_ms=float((int(exit_ts_ns) - position.entry_ts_ns) / 1_000_000.0),
        exit_reason=reason,
    )


def simulate_selective_trades(
    *,
    tick_ts: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    signal_ts: np.ndarray,
    probability_long: np.ndarray,
    threshold: float,
    policy: EconomicPolicy,
    label_settings: LabelSettings | None = None,
) -> tuple[list[ClosedTrade], dict[str, int | float]]:
    """Replay selective signals against executable bid/ask ticks.

    A signal observed at a sample timestamp can enter only on the next tick. Targets and
    stops use the exact same LabelSettings geometry as the historical label builder.
    Favorable target overshoot is not credited; adverse stop gaps are preserved. Positions
    are flattened before a >max_gap market gap and at the end of the supplied slice.
    """
    ts = np.asarray(tick_ts, dtype=np.int64)
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    signal_ts = np.asarray(signal_ts, dtype=np.int64)
    probability_long = np.asarray(probability_long, dtype=float)
    if len(ts) < 2 or bid.shape != ts.shape or ask.shape != ts.shape:
        raise ValueError("aligned tick arrays with at least two rows required")
    if signal_ts.shape != probability_long.shape or not np.isfinite(probability_long).all():
        raise ValueError("aligned finite signal arrays required")
    if not 0.5 <= threshold < 1.0:
        raise ValueError("invalid selective threshold")
    if np.any(np.diff(ts) <= 0):
        raise ValueError("ticks must be strictly increasing")
    if np.any(ask < bid) or not np.isfinite(bid).all() or not np.isfinite(ask).all():
        raise ValueError("invalid executable quotes")

    settings = label_settings or LabelSettings()
    confidence = np.maximum(probability_long, 1.0 - probability_long)
    selected = confidence >= threshold
    selected_ts = signal_ts[selected]
    selected_p = probability_long[selected]
    selected_conf = confidence[selected]

    entry_indices = np.searchsorted(ts, selected_ts, side="right")
    valid = entry_indices < len(ts)
    delay_ns = np.zeros(len(entry_indices), dtype=np.int64)
    delay_ns[valid] = ts[entry_indices[valid]] - selected_ts[valid]
    valid &= delay_ns <= int(policy.max_entry_delay_seconds * 1_000_000_000)

    schedule: dict[int, list[tuple[int, float, float, int]]] = {}
    for seq, (ok, idx, p, conf, sig_ts) in enumerate(
        zip(valid, entry_indices, selected_p, selected_conf, selected_ts), start=1
    ):
        if not bool(ok):
            continue
        schedule.setdefault(int(idx), []).append((seq, float(p), float(conf), int(sig_ts)))

    stats: dict[str, int | float] = {
        "signals_total": int(len(signal_ts)),
        "signals_selected": int(selected.sum()),
        "signals_entry_too_late": int((~valid).sum()),
        "entries_opened": 0,
        "entries_blocked_position_limit": 0,
        "entries_blocked_same_side_limit": 0,
        "entries_blocked_rate_limit": 0,
        "entries_blocked_cooldown": 0,
        "market_gap_flattened": 0,
        "day_end_flattened": 0,
    }

    open_positions: list[OpenTrade] = []
    closed: list[ClosedTrade] = []
    entry_times: deque[int] = deque()
    last_entry_ts: int | None = None
    gap_ns = int(policy.max_gap_seconds * 1_000_000_000)
    cooldown_ns = int(policy.cooldown_ms * 1_000_000)

    for i in range(len(ts)):
        now = int(ts[i])
        survivors: list[OpenTrade] = []
        for position in open_positions:
            if position.side > 0:
                if bid[i] <= position.stop:
                    closed.append(_close_trade(position, exit_price=float(bid[i]), exit_index=i,
                                               exit_ts_ns=now, reason="stop"))
                    continue
                if bid[i] >= position.target:
                    closed.append(_close_trade(position, exit_price=position.target, exit_index=i,
                                               exit_ts_ns=now, reason="target"))
                    continue
            else:
                if ask[i] >= position.stop:
                    closed.append(_close_trade(position, exit_price=float(ask[i]), exit_index=i,
                                               exit_ts_ns=now, reason="stop"))
                    continue
                if ask[i] <= position.target:
                    closed.append(_close_trade(position, exit_price=position.target, exit_index=i,
                                               exit_ts_ns=now, reason="target"))
                    continue
            survivors.append(position)
        open_positions = survivors

        if i + 1 < len(ts) and int(ts[i + 1] - ts[i]) > gap_ns:
            for position in open_positions:
                exit_price = float(bid[i] if position.side > 0 else ask[i])
                closed.append(_close_trade(position, exit_price=exit_price, exit_index=i,
                                           exit_ts_ns=now, reason="market_gap"))
                stats["market_gap_flattened"] = int(stats["market_gap_flattened"]) + 1
            open_positions = []
            continue

        candidates = schedule.get(i)
        if not candidates:
            continue
        for trade_id, p, conf, sig_ts in candidates:
            side = 1 if p >= 0.5 else -1
            while entry_times and now - entry_times[0] >= 1_000_000_000:
                entry_times.popleft()
            if cooldown_ns and last_entry_ts is not None and now - last_entry_ts < cooldown_ns:
                stats["entries_blocked_cooldown"] = int(stats["entries_blocked_cooldown"]) + 1
                continue
            if len(entry_times) >= policy.max_entries_per_second:
                stats["entries_blocked_rate_limit"] = int(stats["entries_blocked_rate_limit"]) + 1
                continue
            if len(open_positions) >= policy.max_positions:
                stats["entries_blocked_position_limit"] = int(stats["entries_blocked_position_limit"]) + 1
                continue
            same_side = sum(1 for position in open_positions if position.side == side)
            if same_side >= policy.max_same_side_positions:
                stats["entries_blocked_same_side_limit"] = int(stats["entries_blocked_same_side_limit"]) + 1
                continue

            spread = max(float(ask[i] - bid[i]), 1e-12)
            if side > 0:
                entry = float(ask[i])
                target = entry + settings.nominal_target_from_entry_spreads * spread
                stop = float(settings.long_stop_price(bid[i], ask[i], spread))
            else:
                entry = float(bid[i])
                target = entry - settings.nominal_target_from_entry_spreads * spread
                stop = float(settings.short_stop_price(bid[i], ask[i], spread))
            open_positions.append(OpenTrade(
                trade_id=int(trade_id),
                signal_ts_ns=int(sig_ts),
                entry_index=i,
                entry_ts_ns=now,
                side=side,
                confidence=float(conf),
                entry=entry,
                entry_spread=spread,
                target=float(target),
                stop=float(stop),
            ))
            entry_times.append(now)
            last_entry_ts = now
            stats["entries_opened"] = int(stats["entries_opened"]) + 1

    if open_positions:
        i = len(ts) - 1
        now = int(ts[i])
        for position in open_positions:
            exit_price = float(bid[i] if position.side > 0 else ask[i])
            closed.append(_close_trade(position, exit_price=exit_price, exit_index=i,
                                       exit_ts_ns=now, reason="day_end"))
            stats["day_end_flattened"] = int(stats["day_end_flattened"]) + 1

    return closed, stats


def economic_metrics(trades: list[ClosedTrade], scenario: CostScenario) -> dict[str, float | int | None]:
    if not trades:
        return {
            "trades": 0,
            "net_pnl_spreads": 0.0,
            "gross_pnl_spreads": 0.0,
            "profit_factor": None,
            "win_rate": None,
            "max_drawdown_spreads": 0.0,
            "average_net_pnl_spreads": None,
            "median_net_pnl_spreads": None,
            "average_hold_ms": None,
            "median_hold_ms": None,
        }
    gross = np.asarray([t.gross_pnl_spreads for t in trades], dtype=float)
    net = gross - scenario.total_extra_cost_spreads
    gains = float(net[net > 0].sum())
    losses = float(-net[net < 0].sum())
    curve = np.cumsum(net)
    peaks = np.maximum.accumulate(np.concatenate(([0.0], curve)))
    curve_with_zero = np.concatenate(([0.0], curve))
    drawdown = peaks - curve_with_zero
    holds = np.asarray([t.hold_ms for t in trades], dtype=float)
    reasons: dict[str, int] = {}
    for trade in trades:
        reasons[trade.exit_reason] = reasons.get(trade.exit_reason, 0) + 1
    return {
        "trades": int(len(trades)),
        "long_trades": int(sum(t.side > 0 for t in trades)),
        "short_trades": int(sum(t.side < 0 for t in trades)),
        "gross_pnl_spreads": float(gross.sum()),
        "net_pnl_spreads": float(net.sum()),
        "average_net_pnl_spreads": float(net.mean()),
        "median_net_pnl_spreads": float(np.median(net)),
        "win_rate": float(np.mean(net > 0)),
        "profit_factor": None if losses <= 1e-12 else float(gains / losses),
        "max_drawdown_spreads": float(drawdown.max()),
        "average_hold_ms": float(holds.mean()),
        "median_hold_ms": float(np.median(holds)),
        "target_exits": int(reasons.get("target", 0)),
        "stop_exits": int(reasons.get("stop", 0)),
        "market_gap_exits": int(reasons.get("market_gap", 0)),
        "day_end_exits": int(reasons.get("day_end", 0)),
        "cost_per_trade_spreads": float(scenario.total_extra_cost_spreads),
    }


def aggregate_economic_metrics(day_metrics: list[dict]) -> dict[str, float | int | None]:
    if not day_metrics:
        return {"days": 0, "trades": 0, "net_pnl_spreads": 0.0}
    trades = int(sum(int(d["trades"]) for d in day_metrics))
    net = float(sum(float(d["net_pnl_spreads"]) for d in day_metrics))
    gross = float(sum(float(d["gross_pnl_spreads"]) for d in day_metrics))
    positive_days = int(sum(float(d["net_pnl_spreads"]) > 0 for d in day_metrics))
    worst_day = float(min(float(d["net_pnl_spreads"]) for d in day_metrics))
    return {
        "days": len(day_metrics),
        "trades": trades,
        "gross_pnl_spreads": gross,
        "net_pnl_spreads": net,
        "average_net_pnl_spreads_per_trade": None if trades == 0 else net / trades,
        "positive_days": positive_days,
        "worst_day_net_pnl_spreads": worst_day,
        "mean_day_net_pnl_spreads": float(np.mean([float(d["net_pnl_spreads"]) for d in day_metrics])),
        "mean_day_profit_factor": (
            None if not any(d.get("profit_factor") is not None for d in day_metrics)
            else float(np.mean([float(d["profit_factor"]) for d in day_metrics if d.get("profit_factor") is not None]))
        ),
        "worst_day_profit_factor": (
            None if not any(d.get("profit_factor") is not None for d in day_metrics)
            else float(min(float(d["profit_factor"]) for d in day_metrics if d.get("profit_factor") is not None))
        ),
    }
