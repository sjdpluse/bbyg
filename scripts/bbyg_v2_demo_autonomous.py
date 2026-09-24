from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from truetrade.scalper.execution import DemoMT5Settings, ExecutionRejected, ExecutionUncertain
from truetrade.scalper.features import TickFeatureEngine
from truetrade.scalper.research_models import RobustScaler, fit_logit
from truetrade.scalper.store import ScalperStore
from truetrade.scalper.timebase import BrokerTimebase, TimeNormalizedDemoMT5Execution
from truetrade.scalper.types import Side, Tick

FEATURE_WINDOW = 96
STRIDE = 4
MAX_HOLD_TICKS = 600
MAX_GAP_NS = 300_000_000_000
MAX_ENTRY_DELAY_NS = 5_000_000_000


@dataclass
class PendingEntry:
    side: Side
    signal_ts_ns: int
    probability_long: float
    confidence: float
    signal_no: int
    copy_no: int


@dataclass
class ActiveTrade:
    ticket: str
    identifier: int
    side: Side
    entry: float
    entry_spread: float
    entry_tick_no: int
    opened_ns: int
    emergency_risk_usd: float
    signal_no: int
    copy_no: int
    peak_favorable_spreads: float = 0.0
    algo_floor_spreads: float | None = None
    protection_stage: str = "none"


def emit(stage: str, **fields) -> None:
    print(json.dumps({"stage": stage, **fields}, sort_keys=True, default=str), flush=True)


def decision_id(kind: str, ts_ns: int, side: Side | None = None) -> str:
    raw = f"v2demo:{kind}:{ts_ns}:{'' if side is None else side.value}:{time.time_ns()}"
    return "v2_" + hashlib.sha256(raw.encode()).hexdigest()[:40]


def load_training(store: ScalperStore, cutoff_ns: int) -> tuple[np.ndarray, np.ndarray]:
    rows = list(store.db.execute(
        """SELECT s.x_json,s.y
           FROM samples s JOIN sample_label_intervals i ON i.feature_ts_ns=s.feature_ts_ns
           WHERE i.label_end_ts_ns < ? ORDER BY s.id DESC LIMIT 20000""",
        (int(cutoff_ns),),
    ))
    rows.reverse()
    if len(rows) < 10_000:
        raise SystemExit(f"insufficient causal training labels: {len(rows)}")
    x = np.asarray([json.loads(r[0]) for r in rows], dtype=float)
    y = np.asarray([int(r[1]) for r in rows], dtype=np.int8)
    if x.shape != (len(rows), 8) or not np.isfinite(x).all():
        raise SystemExit("invalid training matrix")
    return x, y


def wait_closed_outcome(broker: TimeNormalizedDemoMT5Execution, identifier: int):
    outcome = None
    for _ in range(12):
        outcome = broker.closed_outcome(identifier)
        if outcome is not None:
            break
        time.sleep(0.10)
    return outcome


def close_trade(broker: TimeNormalizedDemoMT5Execution, trade: ActiveTrade,
                tick: Tick, reason: str) -> float:
    """Algorithmic market close with broker-side race reconciliation and no blind retries."""
    fill_price = None
    try:
        result = broker.close(trade.ticket, 1.0, decision_id("close", tick.ts_ns, trade.side))
        fill_price = float(result.fill_price)
    except ExecutionRejected:
        current_ticket = broker.position_ticket_by_identifier(trade.identifier)
        if current_ticket is not None:
            raise
        outcome = wait_closed_outcome(broker, trade.identifier)
        if outcome is None:
            raise ExecutionUncertain(
                f"position {trade.ticket} became absent after close rejection but outcome is not observable"
            )
        pnl = float(outcome["net_pnl"])
        emit(
            "closed_reconciled",
            reason=f"{reason}_broker_race",
            side=trade.side.value,
            ticket=trade.ticket,
            fill_price=None,
            net_pnl_usd=pnl,
            peak_favorable_spreads=trade.peak_favorable_spreads,
            algo_floor_spreads=trade.algo_floor_spreads,
            signal_no=trade.signal_no,
            copy_no=trade.copy_no,
        )
        return pnl

    outcome = wait_closed_outcome(broker, trade.identifier)
    if outcome is None:
        raise ExecutionUncertain(f"closed position outcome not observable for {trade.ticket}")
    pnl = float(outcome["net_pnl"])
    emit(
        "closed",
        reason=reason,
        side=trade.side.value,
        ticket=trade.ticket,
        fill_price=fill_price,
        net_pnl_usd=pnl,
        peak_favorable_spreads=trade.peak_favorable_spreads,
        algo_floor_spreads=trade.algo_floor_spreads,
        signal_no=trade.signal_no,
        copy_no=trade.copy_no,
    )
    return pnl


def favorable_spreads(trade: ActiveTrade, tick: Tick) -> float:
    if trade.side is Side.LONG:
        return (tick.bid - trade.entry) / max(trade.entry_spread, 1e-12)
    return (trade.entry - tick.ask) / max(trade.entry_spread, 1e-12)


def directional_probability(side: Side, p_long: float) -> float:
    return p_long if side is Side.LONG else 1.0 - p_long


def update_algorithmic_protection(
    trade: ActiveTrade,
    tick: Tick,
    *,
    break_even_trigger: float,
    profit_lock_trigger: float,
    profit_lock_spreads: float,
    trailing_trigger: float,
    trailing_distance_spreads: float,
) -> None:
    """Software-only profit protection. It does not modify the broker SL."""
    fav = favorable_spreads(trade, tick)
    trade.peak_favorable_spreads = max(trade.peak_favorable_spreads, fav)
    peak = trade.peak_favorable_spreads

    floor = trade.algo_floor_spreads
    stage = trade.protection_stage

    if peak >= break_even_trigger:
        floor = max(0.0, floor if floor is not None else 0.0)
        stage = "break_even"
    if peak >= profit_lock_trigger:
        floor = max(profit_lock_spreads, floor if floor is not None else profit_lock_spreads)
        stage = "profit_lock"
    if peak >= trailing_trigger:
        trailing_floor = peak - trailing_distance_spreads
        floor = max(trailing_floor, floor if floor is not None else trailing_floor)
        stage = "trailing"

    if floor != trade.algo_floor_spreads or stage != trade.protection_stage:
        trade.algo_floor_spreads = floor
        trade.protection_stage = stage
        emit(
            "algo_protection_updated",
            ticket=trade.ticket,
            side=trade.side.value,
            protection=stage,
            peak_favorable_spreads=peak,
            algo_floor_spreads=floor,
            signal_no=trade.signal_no,
            copy_no=trade.copy_no,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXPERIMENTAL multi-position BBYG v2 MT5 DEMO runner with algorithmic exits."
    )
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.52)
    parser.add_argument("--exit-reversal-threshold", type=float, default=0.55)
    parser.add_argument("--edge-fade-threshold", type=float, default=0.52)
    parser.add_argument("--edge-fade-min-profit-spreads", type=float, default=0.35)
    parser.add_argument("--size", type=float, default=0.01)
    parser.add_argument("--entries-per-signal", type=int, default=4)
    parser.add_argument("--max-trades", type=int, default=500)
    parser.add_argument("--max-open-positions", type=int, default=30)
    parser.add_argument("--max-same-side", type=int, default=30)
    parser.add_argument("--max-entries-per-second", type=int, default=8)
    parser.add_argument("--max-session-loss-usd", type=float, default=30.0)
    parser.add_argument("--max-aggregate-emergency-risk-usd", type=float, default=30.0)
    parser.add_argument("--break-even-trigger-spreads", type=float, default=0.65)
    parser.add_argument("--profit-lock-trigger-spreads", type=float, default=0.85)
    parser.add_argument("--profit-lock-spreads", type=float, default=0.20)
    parser.add_argument("--trailing-trigger-spreads", type=float, default=1.00)
    parser.add_argument("--trailing-distance-spreads", type=float, default=0.45)
    parser.add_argument("--poll-ms", type=int, default=10)
    parser.add_argument("--status-seconds", type=float, default=10.0)
    args = parser.parse_args()

    if os.getenv("MT5_MODE", "demo").lower() != "demo":
        raise SystemExit("Refusing to run: MT5_MODE must be demo")
    if os.getenv("BBYG_DEMO_EXECUTION", "false").lower() != "true":
        raise SystemExit("Refusing to trade: BBYG_DEMO_EXECUTION must be true")
    if os.getenv("BBYG_AUTONOMOUS_DEMO_CONFIRM", "") != "I_UNDERSTAND_EXPERIMENTAL_DEMO":
        raise SystemExit(
            "Refusing to trade. Set BBYG_AUTONOMOUS_DEMO_CONFIRM=I_UNDERSTAND_EXPERIMENTAL_DEMO "
            "after confirming the intended MT5 account is DEMO."
        )
    if not 0.50 < args.threshold <= 0.75:
        raise SystemExit("--threshold must be in (0.50, 0.75]")
    if not 0.50 < args.exit_reversal_threshold <= 0.75:
        raise SystemExit("--exit-reversal-threshold must be in (0.50, 0.75]")
    if not 0.50 <= args.edge_fade_threshold <= 0.75:
        raise SystemExit("--edge-fade-threshold must be in [0.50, 0.75]")
    if args.edge_fade_min_profit_spreads < 0:
        raise SystemExit("--edge-fade-min-profit-spreads must be non-negative")
    if not 0 < args.size <= 0.01:
        raise SystemExit("--size must be in (0, 0.01]")
    if not 1 <= args.entries_per_signal <= 10:
        raise SystemExit("--entries-per-signal must be between 1 and 10")
    if not 0.05 <= args.hours <= 8:
        raise SystemExit("--hours must be between 0.05 and 8")
    if not 1 <= args.max_trades <= 2000:
        raise SystemExit("--max-trades must be between 1 and 2000")
    if not 1 <= args.max_open_positions <= 50:
        raise SystemExit("--max-open-positions must be between 1 and 50")
    if not 1 <= args.max_same_side <= args.max_open_positions:
        raise SystemExit("--max-same-side must be between 1 and --max-open-positions")
    if not 1 <= args.max_entries_per_second <= 20:
        raise SystemExit("--max-entries-per-second must be between 1 and 20")
    if not 0 < args.max_session_loss_usd <= 100:
        raise SystemExit("--max-session-loss-usd must be in (0, 100]")
    if not 0 < args.max_aggregate_emergency_risk_usd <= 100:
        raise SystemExit("--max-aggregate-emergency-risk-usd must be in (0, 100]")
    if not 0 < args.break_even_trigger_spreads < args.profit_lock_trigger_spreads < args.trailing_trigger_spreads:
        raise SystemExit("Require break-even < profit-lock < trailing trigger")
    if not 0 <= args.profit_lock_spreads < args.profit_lock_trigger_spreads:
        raise SystemExit("invalid profit lock")
    if not 0 < args.trailing_distance_spreads < args.trailing_trigger_spreads:
        raise SystemExit("invalid trailing distance")

    settings = DemoMT5Settings.from_env()
    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-multiday-v1"))
    store = ScalperStore(state_dir / "scalper.sqlite")
    try:
        train_x, train_y = load_training(store, time.time_ns())
    finally:
        store.close()

    scaler = RobustScaler.fit(train_x)
    model = fit_logit(scaler.transform(train_x), train_y, iterations=140, balanced=True)

    broker = TimeNormalizedDemoMT5Execution(settings, timebase=BrokerTimebase.from_env())
    feature_engine = TickFeatureEngine(window=FEATURE_WINDOW, min_ticks=FEATURE_WINDOW,
                                       fast_ticks=8, slow_ticks=24)
    pending: deque[PendingEntry] = deque()
    active: dict[str, ActiveTrade] = {}
    entry_times: deque[int] = deque()
    last_tick: Tick | None = None
    tick_no = 0
    segment_tick_no = 0
    anchors_evaluated = 0
    signals = 0
    trades_opened = 0
    trades_closed = 0
    realized_pnl = 0.0
    last_p_long: float | None = None
    last_confidence: float | None = None
    max_confidence_seen = 0.0
    last_status_monotonic = 0.0
    deadline = time.monotonic() + args.hours * 3600.0
    uncertain_state = False
    start_equity: float | None = None

    try:
        broker.connect()
        account = broker._account(trading=True)
        start_equity = float(account.equity)
        if broker.positions():
            raise SystemExit("Refusing to start: existing BBYG position detected")
        emit(
            "started",
            experimental=True,
            demo_only=True,
            multi_position=True,
            algorithmic_exits=True,
            broker_stop_role="emergency_fail_safe_only",
            login=int(account.login),
            server=str(account.server),
            symbol=broker.symbol,
            threshold=args.threshold,
            exit_reversal_threshold=args.exit_reversal_threshold,
            entries_per_signal=args.entries_per_signal,
            size=args.size,
            max_trades=args.max_trades,
            max_open_positions=args.max_open_positions,
            max_same_side=args.max_same_side,
            max_entries_per_second=args.max_entries_per_second,
            max_session_loss_usd=args.max_session_loss_usd,
            max_aggregate_emergency_risk_usd=args.max_aggregate_emergency_risk_usd,
            training_samples=len(train_x),
            training_long_fraction=float(np.mean(train_y)),
        )

        while time.monotonic() < deadline:
            current_equity = broker.account_equity()
            if current_equity <= start_equity - args.max_session_loss_usd:
                emit(
                    "session_stop",
                    reason="max_equity_drawdown",
                    start_equity=start_equity,
                    current_equity=current_equity,
                    drawdown_usd=start_equity - current_equity,
                )
                break
            if trades_opened >= args.max_trades and not active:
                emit("session_stop", reason="max_trades", trades_opened=trades_opened)
                break

            tick = broker.latest_tick()
            if tick is None:
                time.sleep(args.poll_ms / 1000.0)
                continue

            if last_tick is not None and tick.ts_ns - last_tick.ts_ns > MAX_GAP_NS:
                pending.clear()
                feature_engine = TickFeatureEngine(window=FEATURE_WINDOW, min_ticks=FEATURE_WINDOW,
                                                   fast_ticks=8, slow_ticks=24)
                segment_tick_no = 0
                emit("market_gap_reset", open_positions=len(active))

            tick_no += 1
            segment_tick_no += 1

            # Reconcile broker-side closures before any new write.
            for ticket in list(active):
                trade = active.get(ticket)
                if trade is None:
                    continue
                current_ticket = broker.position_ticket_by_identifier(trade.identifier)
                if current_ticket is None:
                    outcome = wait_closed_outcome(broker, trade.identifier)
                    if outcome is None:
                        raise ExecutionUncertain(
                            f"managed position disappeared without observable outcome: {ticket}"
                        )
                    active.pop(ticket, None)
                    pnl = float(outcome["net_pnl"])
                    realized_pnl += pnl
                    trades_closed += 1
                    emit(
                        "broker_emergency_closed",
                        ticket=ticket,
                        side=trade.side.value,
                        net_pnl_usd=pnl,
                        session_net_pnl_usd=realized_pnl,
                        outcome=outcome,
                    )
                elif current_ticket != ticket:
                    active.pop(ticket, None)
                    trade.ticket = current_ticket
                    active[current_ticket] = trade
                    emit("position_ticket_reconciled", old_ticket=ticket,
                         new_ticket=current_ticket, identifier=trade.identifier)

            # Build/update the causal model score before managing algorithmic exits.
            new_model_score = False
            features = feature_engine.update(tick)
            if features is not None and (segment_tick_no - FEATURE_WINDOW) % STRIDE == 0:
                anchors_evaluated += 1
                p_long = float(model.probability(scaler.transform(
                    np.asarray([features.vector()], dtype=float)
                ))[0])
                confidence = max(p_long, 1.0 - p_long)
                last_p_long = p_long
                last_confidence = confidence
                max_confidence_seen = max(max_confidence_seen, confidence)
                new_model_score = True

            # Software-only profit protection and model-driven exits.
            for ticket in list(active):
                trade = active.get(ticket)
                if trade is None:
                    continue
                update_algorithmic_protection(
                    trade,
                    tick,
                    break_even_trigger=args.break_even_trigger_spreads,
                    profit_lock_trigger=args.profit_lock_trigger_spreads,
                    profit_lock_spreads=args.profit_lock_spreads,
                    trailing_trigger=args.trailing_trigger_spreads,
                    trailing_distance_spreads=args.trailing_distance_spreads,
                )
                fav = favorable_spreads(trade, tick)
                reason = None

                if trade.algo_floor_spreads is not None and fav <= trade.algo_floor_spreads:
                    reason = "algorithmic_profit_protection"
                elif new_model_score and last_p_long is not None:
                    dir_p = directional_probability(trade.side, last_p_long)
                    opposite_p = 1.0 - dir_p
                    if opposite_p >= args.exit_reversal_threshold:
                        reason = "model_reversal"
                    elif fav >= args.edge_fade_min_profit_spreads and dir_p < args.edge_fade_threshold:
                        reason = "profitable_edge_fade"
                if reason is None and tick_no - trade.entry_tick_no >= MAX_HOLD_TICKS:
                    reason = "algorithmic_horizon"

                if reason is not None:
                    pnl = close_trade(broker, trade, tick, reason)
                    active.pop(trade.ticket, None)
                    active.pop(ticket, None)
                    realized_pnl += pnl
                    trades_closed += 1

            aggregate_emergency_risk = float(sum(t.emergency_risk_usd for t in active.values()))
            while entry_times and tick.ts_ns - entry_times[0] >= 1_000_000_000:
                entry_times.popleft()

            # A selected score creates N independent tickets (fanout), all still bounded by risk/rate/portfolio caps.
            if new_model_score and last_p_long is not None and last_confidence is not None:
                if last_confidence >= args.threshold and trades_opened < args.max_trades:
                    side = Side.LONG if last_p_long >= 0.5 else Side.SHORT
                    signals += 1
                    for copy_no in range(1, args.entries_per_signal + 1):
                        pending.append(PendingEntry(
                            side=side,
                            signal_ts_ns=tick.ts_ns,
                            probability_long=last_p_long,
                            confidence=last_confidence,
                            signal_no=signals,
                            copy_no=copy_no,
                        ))
                    emit(
                        "signal",
                        side=side.value,
                        probability_long=last_p_long,
                        confidence=last_confidence,
                        threshold=args.threshold,
                        signal_no=signals,
                        fanout=args.entries_per_signal,
                        queued_signals=len(pending),
                        open_positions=len(active),
                    )

            # Execute queued entries on strictly later ticks. Rate-limited items are carried, not discarded.
            carry: deque[PendingEntry] = deque()
            queued = len(pending)
            for _ in range(queued):
                signal = pending.popleft()
                delay_ns = tick.ts_ns - signal.signal_ts_ns
                if delay_ns <= 0:
                    carry.append(signal)
                    continue
                if delay_ns > MAX_ENTRY_DELAY_NS or trades_opened >= args.max_trades:
                    continue
                if len(active) >= args.max_open_positions:
                    carry.append(signal)
                    continue
                if sum(1 for t in active.values() if t.side is signal.side) >= args.max_same_side:
                    carry.append(signal)
                    continue
                while entry_times and tick.ts_ns - entry_times[0] >= 1_000_000_000:
                    entry_times.popleft()
                if len(entry_times) >= args.max_entries_per_second:
                    carry.append(signal)
                    continue

                info = broker._symbol_info()
                expected = tick.ask if signal.side is Side.LONG else tick.bid
                _, candidate_risk = broker._risk_checked_stop(signal.side, args.size, expected, info)
                aggregate_emergency_risk = float(sum(t.emergency_risk_usd for t in active.values()))
                if aggregate_emergency_risk + candidate_risk > args.max_aggregate_emergency_risk_usd + 1e-9:
                    carry.append(signal)
                    continue

                opened = broker.open(signal.side, args.size, decision_id("open", tick.ts_ns, signal.side))
                if opened.position_id is None or opened.position_identifier is None:
                    raise ExecutionUncertain("open result missing position identity")
                positions = broker.positions()
                pos = next((p for p in positions if p.position_id == opened.position_id), None)
                if pos is None:
                    raise ExecutionUncertain("opened position not observable")
                spread = max(tick.spread, 1e-12)
                entry = float(pos.entry)
                trade = ActiveTrade(
                    ticket=opened.position_id,
                    identifier=int(opened.position_identifier),
                    side=signal.side,
                    entry=entry,
                    entry_spread=spread,
                    entry_tick_no=tick_no,
                    opened_ns=tick.ts_ns,
                    emergency_risk_usd=float(opened.risk_amount or candidate_risk),
                    signal_no=signal.signal_no,
                    copy_no=signal.copy_no,
                )
                active[trade.ticket] = trade
                trades_opened += 1
                entry_times.append(tick.ts_ns)
                emit(
                    "opened",
                    side=signal.side.value,
                    confidence=signal.confidence,
                    probability_long=signal.probability_long,
                    signal_no=signal.signal_no,
                    copy_no=signal.copy_no,
                    ticket=trade.ticket,
                    entry=entry,
                    broker_emergency_stop=pos.broker_stop,
                    emergency_broker_stop_risk_usd=trade.emergency_risk_usd,
                    aggregate_emergency_risk_usd=sum(t.emergency_risk_usd for t in active.values()),
                    open_positions=len(active),
                    trades_opened=trades_opened,
                )
            pending = carry

            now = time.monotonic()
            if now - last_status_monotonic >= args.status_seconds:
                emit(
                    "status",
                    ticks=tick_no,
                    anchors_evaluated=anchors_evaluated,
                    last_probability_long=last_p_long,
                    last_confidence=last_confidence,
                    max_confidence_seen=max_confidence_seen,
                    threshold=args.threshold,
                    signals=signals,
                    queued_entries=len(pending),
                    open_positions=len(active),
                    long_positions=sum(1 for t in active.values() if t.side is Side.LONG),
                    short_positions=sum(1 for t in active.values() if t.side is Side.SHORT),
                    protected_positions=sum(1 for t in active.values() if t.algo_floor_spreads is not None),
                    trailing_positions=sum(1 for t in active.values() if t.protection_stage == "trailing"),
                    trades_opened=trades_opened,
                    trades_closed=trades_closed,
                    realized_pnl_usd=realized_pnl,
                    equity=broker.account_equity(),
                    aggregate_emergency_risk_usd=sum(t.emergency_risk_usd for t in active.values()),
                )
                last_status_monotonic = now

            last_tick = tick
            time.sleep(args.poll_ms / 1000.0)

    except ExecutionRejected as exc:
        emit("execution_rejected", error=str(exc))
        raise SystemExit(2) from None
    except ExecutionUncertain as exc:
        uncertain_state = True
        emit("execution_uncertain", error=str(exc), instruction="Inspect MT5 before any retry")
        raise SystemExit(3) from None
    finally:
        if broker.connected and not uncertain_state:
            try:
                fresh = broker._fresh_tick_for_write()
                for ticket in list(active):
                    trade = active.get(ticket)
                    if trade is None:
                        continue
                    try:
                        pnl = close_trade(broker, trade, fresh, "session_end")
                        active.pop(ticket, None)
                        active.pop(trade.ticket, None)
                        realized_pnl += pnl
                        trades_closed += 1
                    except ExecutionRejected as exc:
                        emit("final_flatten_rejected", ticket=ticket, error=str(exc),
                             instruction="Inspect MT5 manually before restarting")
                    except ExecutionUncertain as exc:
                        uncertain_state = True
                        emit("final_flatten_uncertain", ticket=ticket, error=str(exc),
                             instruction="Inspect MT5 manually before restarting")
                        break
            finally:
                broker.shutdown()
        elif broker.connected:
            broker.shutdown()

    emit(
        "complete",
        demo_only=True,
        experimental=True,
        multi_position=True,
        algorithmic_exits=True,
        signals=signals,
        anchors_evaluated=anchors_evaluated,
        max_confidence_seen=max_confidence_seen,
        trades_opened=trades_opened,
        trades_closed=trades_closed,
        remaining_open_positions=len(active),
        session_net_pnl_usd=realized_pnl,
        uncertain_state=uncertain_state,
    )


if __name__ == "__main__":
    main()
