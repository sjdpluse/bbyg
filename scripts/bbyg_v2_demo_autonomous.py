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
TARGET_SPREADS = 1.8
INITIAL_STOP_SPREADS = 1.4
MAX_HOLD_TICKS = 600
MAX_GAP_NS = 300_000_000_000
MAX_ENTRY_DELAY_NS = 5_000_000_000


@dataclass
class PendingEntry:
    side: Side
    signal_ts_ns: int
    probability_long: float
    confidence: float


@dataclass
class ActiveTrade:
    ticket: str
    identifier: int
    side: Side
    entry: float
    entry_spread: float
    target: float
    stop: float
    entry_tick_no: int
    opened_ns: int
    protected_stop: float | None = None
    protection_stage: int = 0
    peak_favorable_spreads: float = 0.0


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
    """Close one managed trade while tolerating a broker-SL race without blind retries."""
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
            outcome=outcome,
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
    )
    return pnl


def favorable_spreads(trade: ActiveTrade, tick: Tick) -> float:
    if trade.side is Side.LONG:
        return (tick.bid - trade.entry) / max(trade.entry_spread, 1e-12)
    return (trade.entry - tick.ask) / max(trade.entry_spread, 1e-12)


def maybe_tighten_profit_stop(
    broker: TimeNormalizedDemoMT5Execution,
    trade: ActiveTrade,
    tick: Tick,
    *,
    break_even_trigger: float,
    profit_lock_trigger: float,
    profit_lock_spreads: float,
    trailing_trigger: float,
    trailing_distance_spreads: float,
    protection_step_spreads: float,
) -> None:
    """Move broker SL only in the profitable direction: break-even -> lock -> trail."""
    fav = favorable_spreads(trade, tick)
    trade.peak_favorable_spreads = max(trade.peak_favorable_spreads, fav)
    desired = None
    desired_stage = trade.protection_stage
    stage_name = None

    if fav >= break_even_trigger and trade.protection_stage < 1:
        desired = trade.entry
        desired_stage = 1
        stage_name = "break_even"

    if fav >= profit_lock_trigger:
        locked = (trade.entry + profit_lock_spreads * trade.entry_spread
                  if trade.side is Side.LONG
                  else trade.entry - profit_lock_spreads * trade.entry_spread)
        if desired is None or (trade.side is Side.LONG and locked > desired) or (
            trade.side is Side.SHORT and locked < desired
        ):
            desired = locked
            desired_stage = max(desired_stage, 2)
            stage_name = "profit_lock"

    if fav >= trailing_trigger:
        trailing = (tick.bid - trailing_distance_spreads * trade.entry_spread
                    if trade.side is Side.LONG
                    else tick.ask + trailing_distance_spreads * trade.entry_spread)
        if desired is None or (trade.side is Side.LONG and trailing > desired) or (
            trade.side is Side.SHORT and trailing < desired
        ):
            desired = trailing
            desired_stage = 3
            stage_name = "trailing"

    if desired is None:
        return

    current = trade.protected_stop
    min_improvement = protection_step_spreads * trade.entry_spread
    if current is not None:
        if trade.side is Side.LONG and desired <= current + min_improvement:
            return
        if trade.side is Side.SHORT and desired >= current - min_improvement:
            return

    try:
        result = broker.tighten_stop(
            trade.ticket,
            float(desired),
            decision_id("protect", tick.ts_ns, trade.side),
        )
    except ExecutionRejected as exc:
        # A temporary freeze/min-distance rejection is non-fatal because the existing
        # emergency SL stays on the broker. If the position vanished, reconciliation
        # on the next tick will collect its closed outcome.
        emit(
            "protection_rejected",
            ticket=trade.ticket,
            side=trade.side.value,
            protection=stage_name,
            favorable_spreads=fav,
            error=str(exc),
        )
        return

    actual_stop = float(result.fill_price)
    trade.protected_stop = actual_stop
    trade.protection_stage = desired_stage
    if trade.side is Side.LONG:
        trade.stop = max(trade.stop, actual_stop)
    else:
        trade.stop = min(trade.stop, actual_stop)
    emit(
        "protection_tightened",
        ticket=trade.ticket,
        side=trade.side.value,
        protection=stage_name,
        favorable_spreads=fav,
        peak_favorable_spreads=trade.peak_favorable_spreads,
        new_stop=actual_stop,
        entry=trade.entry,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXPERIMENTAL multi-position BBYG v2 MT5 DEMO runner. Never permits a live account."
    )
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--size", type=float, default=0.01)
    parser.add_argument("--max-trades", type=int, default=300)
    parser.add_argument("--max-open-positions", type=int, default=30)
    parser.add_argument("--max-same-side", type=int, default=24)
    parser.add_argument("--max-entries-per-second", type=int, default=4)
    parser.add_argument("--max-session-loss-usd", type=float, default=20.0)
    parser.add_argument("--max-aggregate-emergency-risk-usd", type=float, default=15.0)
    parser.add_argument("--break-even-trigger-spreads", type=float, default=0.80)
    parser.add_argument("--profit-lock-trigger-spreads", type=float, default=1.00)
    parser.add_argument("--profit-lock-spreads", type=float, default=0.25)
    parser.add_argument("--trailing-trigger-spreads", type=float, default=1.20)
    parser.add_argument("--trailing-distance-spreads", type=float, default=0.60)
    parser.add_argument("--protection-step-spreads", type=float, default=0.10)
    parser.add_argument("--poll-ms", type=int, default=10)
    parser.add_argument("--status-seconds", type=float, default=15.0)
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
    if not 0.51 <= args.threshold <= 0.75:
        raise SystemExit("--threshold must be between 0.51 and 0.75")
    if not 0 < args.size <= 0.01:
        raise SystemExit("--size must be in (0, 0.01]")
    if not 0.05 <= args.hours <= 8:
        raise SystemExit("--hours must be between 0.05 and 8")
    if not 1 <= args.max_trades <= 1000:
        raise SystemExit("--max-trades must be between 1 and 1000")
    if not 1 <= args.max_open_positions <= 50:
        raise SystemExit("--max-open-positions must be between 1 and 50")
    if not 1 <= args.max_same_side <= args.max_open_positions:
        raise SystemExit("--max-same-side must be between 1 and --max-open-positions")
    if not 1 <= args.max_entries_per_second <= 10:
        raise SystemExit("--max-entries-per-second must be between 1 and 10")
    if not 0 < args.max_session_loss_usd <= 50:
        raise SystemExit("--max-session-loss-usd must be in (0, 50]")
    if not 0 < args.max_aggregate_emergency_risk_usd <= 25:
        raise SystemExit("--max-aggregate-emergency-risk-usd must be in (0, 25]")
    if not 0 < args.break_even_trigger_spreads < args.profit_lock_trigger_spreads:
        raise SystemExit("break-even trigger must be positive and below profit-lock trigger")
    if not args.break_even_trigger_spreads < args.profit_lock_trigger_spreads < args.trailing_trigger_spreads:
        raise SystemExit("protection triggers must increase: break-even < profit-lock < trailing")
    if not 0 <= args.profit_lock_spreads < args.profit_lock_trigger_spreads:
        raise SystemExit("profit-lock spreads must be non-negative and below its trigger")
    if not 0 < args.trailing_distance_spreads < args.trailing_trigger_spreads:
        raise SystemExit("trailing distance must be positive and below trailing trigger")
    if not 0 < args.protection_step_spreads <= 0.5:
        raise SystemExit("protection step must be in (0, 0.5]")
    if not 5 <= args.poll_ms <= 1000:
        raise SystemExit("--poll-ms must be between 5 and 1000")
    if args.status_seconds <= 0:
        raise SystemExit("--status-seconds must be positive")

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
    aggregate_emergency_risk = 0.0
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
            profit_protection=True,
            login=int(account.login),
            server=str(account.server),
            symbol=broker.symbol,
            threshold=args.threshold,
            size=args.size,
            max_trades=args.max_trades,
            max_open_positions=args.max_open_positions,
            max_same_side=args.max_same_side,
            max_entries_per_second=args.max_entries_per_second,
            max_session_loss_usd=args.max_session_loss_usd,
            max_aggregate_emergency_risk_usd=args.max_aggregate_emergency_risk_usd,
            break_even_trigger_spreads=args.break_even_trigger_spreads,
            profit_lock_trigger_spreads=args.profit_lock_trigger_spreads,
            profit_lock_spreads=args.profit_lock_spreads,
            trailing_trigger_spreads=args.trailing_trigger_spreads,
            trailing_distance_spreads=args.trailing_distance_spreads,
            start_equity=start_equity,
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

            # Reconcile every managed position by stable identifier before new writes.
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
                        "broker_closed",
                        ticket=ticket,
                        side=trade.side.value,
                        net_pnl_usd=pnl,
                        session_net_pnl_usd=realized_pnl,
                        protection_stage=trade.protection_stage,
                        peak_favorable_spreads=trade.peak_favorable_spreads,
                        outcome=outcome,
                    )
                elif current_ticket != ticket:
                    active.pop(ticket, None)
                    trade.ticket = current_ticket
                    active[current_ticket] = trade
                    emit("position_ticket_reconciled", old_ticket=ticket,
                         new_ticket=current_ticket, identifier=trade.identifier)

            # First protect profitable positions, then evaluate exits independently.
            for ticket in list(active):
                trade = active.get(ticket)
                if trade is None:
                    continue
                maybe_tighten_profit_stop(
                    broker,
                    trade,
                    tick,
                    break_even_trigger=args.break_even_trigger_spreads,
                    profit_lock_trigger=args.profit_lock_trigger_spreads,
                    profit_lock_spreads=args.profit_lock_spreads,
                    trailing_trigger=args.trailing_trigger_spreads,
                    trailing_distance_spreads=args.trailing_distance_spreads,
                    protection_step_spreads=args.protection_step_spreads,
                )

            for ticket in list(active):
                trade = active.get(ticket)
                if trade is None:
                    continue
                reason = None
                if trade.side is Side.LONG:
                    if tick.bid <= trade.stop:
                        reason = "protected_stop" if trade.protection_stage else "strategy_stop"
                    elif tick.bid >= trade.target:
                        reason = "strategy_target"
                else:
                    if tick.ask >= trade.stop:
                        reason = "protected_stop" if trade.protection_stage else "strategy_stop"
                    elif tick.ask <= trade.target:
                        reason = "strategy_target"
                if reason is None and tick_no - trade.entry_tick_no >= MAX_HOLD_TICKS:
                    reason = "strategy_horizon"
                if reason is not None:
                    pnl = close_trade(broker, trade, tick, reason)
                    active.pop(trade.ticket, None)
                    active.pop(ticket, None)
                    realized_pnl += pnl
                    trades_closed += 1

            # Conservative aggregate emergency-stop risk. Tightened stops may make true risk lower.
            if active:
                info = broker._symbol_info()
                risks = []
                for trade in active.values():
                    expected = tick.ask if trade.side is Side.LONG else tick.bid
                    _, risk = broker._risk_checked_stop(trade.side, args.size, expected, info)
                    risks.append(float(risk))
                aggregate_emergency_risk = float(sum(risks))
            else:
                aggregate_emergency_risk = 0.0

            while entry_times and tick.ts_ns - entry_times[0] >= 1_000_000_000:
                entry_times.popleft()

            # Every selected model signal may create its own independent position.
            queued = len(pending)
            for _ in range(queued):
                signal = pending.popleft()
                delay_ns = tick.ts_ns - signal.signal_ts_ns
                if delay_ns <= 0:
                    pending.append(signal)
                    continue
                if delay_ns > MAX_ENTRY_DELAY_NS or trades_opened >= args.max_trades:
                    continue
                if len(active) >= args.max_open_positions:
                    continue
                if sum(1 for t in active.values() if t.side is signal.side) >= args.max_same_side:
                    continue
                if len(entry_times) >= args.max_entries_per_second:
                    continue

                info = broker._symbol_info()
                expected = tick.ask if signal.side is Side.LONG else tick.bid
                _, candidate_risk = broker._risk_checked_stop(signal.side, args.size, expected, info)
                if aggregate_emergency_risk + candidate_risk > args.max_aggregate_emergency_risk_usd + 1e-9:
                    emit(
                        "entry_blocked",
                        reason="aggregate_risk_cap",
                        open_positions=len(active),
                        aggregate_risk_usd=aggregate_emergency_risk,
                        candidate_risk_usd=candidate_risk,
                    )
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
                if signal.side is Side.LONG:
                    target = entry + TARGET_SPREADS * spread
                    stop = entry - INITIAL_STOP_SPREADS * spread
                else:
                    target = entry - TARGET_SPREADS * spread
                    stop = entry + INITIAL_STOP_SPREADS * spread
                active[opened.position_id] = ActiveTrade(
                    ticket=opened.position_id,
                    identifier=int(opened.position_identifier),
                    side=signal.side,
                    entry=entry,
                    entry_spread=spread,
                    target=float(target),
                    stop=float(stop),
                    entry_tick_no=tick_no,
                    opened_ns=tick.ts_ns,
                    protected_stop=pos.broker_stop,
                )
                trades_opened += 1
                entry_times.append(tick.ts_ns)
                aggregate_emergency_risk += float(opened.risk_amount or candidate_risk)
                emit(
                    "opened",
                    side=signal.side.value,
                    confidence=signal.confidence,
                    probability_long=signal.probability_long,
                    ticket=opened.position_id,
                    entry=entry,
                    target=target,
                    strategy_stop=stop,
                    broker_emergency_stop=pos.broker_stop,
                    emergency_broker_stop_risk_usd=opened.risk_amount,
                    aggregate_emergency_risk_usd=aggregate_emergency_risk,
                    open_positions=len(active),
                    trades_opened=trades_opened,
                )

            features = feature_engine.update(tick)
            if features is not None and trades_opened < args.max_trades:
                if (segment_tick_no - FEATURE_WINDOW) % STRIDE == 0:
                    anchors_evaluated += 1
                    p_long = float(model.probability(scaler.transform(
                        np.asarray([features.vector()], dtype=float)
                    ))[0])
                    confidence = max(p_long, 1.0 - p_long)
                    last_p_long = p_long
                    last_confidence = confidence
                    max_confidence_seen = max(max_confidence_seen, confidence)
                    if confidence >= args.threshold:
                        side = Side.LONG if p_long >= 0.5 else Side.SHORT
                        pending.append(PendingEntry(side, tick.ts_ns, p_long, confidence))
                        signals += 1
                        emit(
                            "signal",
                            side=side.value,
                            probability_long=p_long,
                            confidence=confidence,
                            threshold=args.threshold,
                            signals=signals,
                            queued_signals=len(pending),
                            open_positions=len(active),
                        )

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
                    queued_signals=len(pending),
                    open_positions=len(active),
                    long_positions=sum(1 for t in active.values() if t.side is Side.LONG),
                    short_positions=sum(1 for t in active.values() if t.side is Side.SHORT),
                    protected_positions=sum(1 for t in active.values() if t.protection_stage > 0),
                    trailing_positions=sum(1 for t in active.values() if t.protection_stage >= 3),
                    trades_opened=trades_opened,
                    trades_closed=trades_closed,
                    realized_pnl_usd=realized_pnl,
                    equity=broker.account_equity(),
                    aggregate_emergency_risk_usd=aggregate_emergency_risk,
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
        profit_protection=True,
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
