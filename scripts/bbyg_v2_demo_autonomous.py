from __future__ import annotations

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
STOP_SPREADS = 1.4
MAX_HOLD_TICKS = 600
MAX_GAP_NS = 300_000_000_000


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXPERIMENTAL autonomous BBYG v2 MT5 DEMO runner. Never permits a live account."
    )
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--size", type=float, default=0.01)
    parser.add_argument("--max-trades", type=int, default=20)
    parser.add_argument("--max-session-loss-usd", type=float, default=5.0)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--poll-ms", type=int, default=10)
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
    if not 0.55 <= args.threshold <= 0.75:
        raise SystemExit("--threshold must be between 0.55 and 0.75")
    if not 0 < args.size <= 0.01:
        raise SystemExit("--size must be in (0, 0.01]")
    if not 0.05 <= args.hours <= 8:
        raise SystemExit("--hours must be between 0.05 and 8")
    if not 1 <= args.max_trades <= 100:
        raise SystemExit("--max-trades must be between 1 and 100")
    if not 0 < args.max_session_loss_usd <= 20:
        raise SystemExit("--max-session-loss-usd must be in (0, 20]")
    if not 0 <= args.cooldown_seconds <= 300:
        raise SystemExit("--cooldown-seconds must be between 0 and 300")

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
    pending: PendingEntry | None = None
    active: ActiveTrade | None = None
    last_tick: Tick | None = None
    tick_no = 0
    segment_tick_no = 0
    signals = 0
    trades_opened = 0
    trades_closed = 0
    realized_pnl = 0.0
    last_close_monotonic = -1e30
    deadline = time.monotonic() + args.hours * 3600.0

    try:
        broker.connect()
        account = broker._account(trading=True)
        if broker.positions():
            raise SystemExit("Refusing to start: existing BBYG position detected")
        emit(
            "started",
            experimental=True,
            demo_only=True,
            login=int(account.login),
            server=str(account.server),
            symbol=broker.symbol,
            threshold=args.threshold,
            size=args.size,
            max_trades=args.max_trades,
            max_session_loss_usd=args.max_session_loss_usd,
            cooldown_seconds=args.cooldown_seconds,
            training_samples=len(train_x),
            training_long_fraction=float(np.mean(train_y)),
        )

        while time.monotonic() < deadline:
            if trades_opened >= args.max_trades:
                emit("session_stop", reason="max_trades", trades_opened=trades_opened)
                break
            if realized_pnl <= -args.max_session_loss_usd:
                emit("session_stop", reason="max_session_loss", realized_pnl=realized_pnl)
                break

            tick = broker.latest_tick()
            if tick is None:
                time.sleep(args.poll_ms / 1000.0)
                continue

            if last_tick is not None and tick.ts_ns - last_tick.ts_ns > MAX_GAP_NS:
                pending = None
                feature_engine = TickFeatureEngine(window=FEATURE_WINDOW, min_ticks=FEATURE_WINDOW,
                                                   fast_ticks=8, slow_ticks=24)
                segment_tick_no = 0
                if active is not None:
                    emit("market_gap_with_open_position", ticket=active.ticket)

            tick_no += 1
            segment_tick_no += 1

            if active is not None:
                exit_reason = None
                if active.side is Side.LONG:
                    if tick.bid <= active.stop:
                        exit_reason = "strategy_stop"
                    elif tick.bid >= active.target:
                        exit_reason = "strategy_target"
                else:
                    if tick.ask >= active.stop:
                        exit_reason = "strategy_stop"
                    elif tick.ask <= active.target:
                        exit_reason = "strategy_target"
                if exit_reason is None and tick_no - active.entry_tick_no >= MAX_HOLD_TICKS:
                    exit_reason = "strategy_horizon"

                if exit_reason is not None:
                    result = broker.close(active.ticket, 1.0, decision_id("close", tick.ts_ns, active.side))
                    outcome = None
                    for _ in range(12):
                        outcome = broker.closed_outcome(active.identifier)
                        if outcome is not None:
                            break
                        time.sleep(0.10)
                    if outcome is None:
                        raise ExecutionUncertain("closed position outcome not observable")
                    pnl = float(outcome["net_pnl"])
                    realized_pnl += pnl
                    trades_closed += 1
                    emit(
                        "closed",
                        reason=exit_reason,
                        side=active.side.value,
                        ticket=active.ticket,
                        fill_price=result.fill_price,
                        net_pnl_usd=pnl,
                        session_net_pnl_usd=realized_pnl,
                        trades_closed=trades_closed,
                    )
                    active = None
                    pending = None
                    last_close_monotonic = time.monotonic()

            if active is None and pending is not None:
                delay_ns = tick.ts_ns - pending.signal_ts_ns
                if 0 < delay_ns <= 5_000_000_000:
                    if time.monotonic() - last_close_monotonic >= args.cooldown_seconds:
                        opened = broker.open(pending.side, args.size,
                                             decision_id("open", tick.ts_ns, pending.side))
                        if opened.position_id is None or opened.position_identifier is None:
                            raise ExecutionUncertain("open result missing position identity")
                        positions = broker.positions()
                        pos = next((p for p in positions if p.position_id == opened.position_id), None)
                        if pos is None:
                            raise ExecutionUncertain("opened position not observable")
                        spread = max(tick.spread, 1e-12)
                        entry = float(pos.entry)
                        if pending.side is Side.LONG:
                            target = entry + TARGET_SPREADS * spread
                            stop = entry - STOP_SPREADS * spread
                        else:
                            target = entry - TARGET_SPREADS * spread
                            stop = entry + STOP_SPREADS * spread
                        active = ActiveTrade(
                            ticket=opened.position_id,
                            identifier=int(opened.position_identifier),
                            side=pending.side,
                            entry=entry,
                            entry_spread=spread,
                            target=float(target),
                            stop=float(stop),
                            entry_tick_no=tick_no,
                            opened_ns=tick.ts_ns,
                        )
                        trades_opened += 1
                        emit(
                            "opened",
                            side=pending.side.value,
                            confidence=pending.confidence,
                            probability_long=pending.probability_long,
                            ticket=opened.position_id,
                            entry=entry,
                            target=target,
                            strategy_stop=stop,
                            emergency_broker_stop_risk_usd=opened.risk_amount,
                            trades_opened=trades_opened,
                        )
                    pending = None
                elif delay_ns > 5_000_000_000:
                    pending = None

            features = feature_engine.update(tick)
            if active is None and pending is None and features is not None:
                if (segment_tick_no - FEATURE_WINDOW) % STRIDE == 0:
                    p_long = float(model.probability(scaler.transform(
                        np.asarray([features.vector()], dtype=float)
                    ))[0])
                    confidence = max(p_long, 1.0 - p_long)
                    if confidence >= args.threshold:
                        side = Side.LONG if p_long >= 0.5 else Side.SHORT
                        pending = PendingEntry(side, tick.ts_ns, p_long, confidence)
                        signals += 1
                        emit(
                            "signal",
                            side=side.value,
                            probability_long=p_long,
                            confidence=confidence,
                            threshold=args.threshold,
                            signals=signals,
                        )

            last_tick = tick
            time.sleep(args.poll_ms / 1000.0)

    except ExecutionRejected as exc:
        emit("execution_rejected", error=str(exc))
        raise SystemExit(2) from None
    except ExecutionUncertain as exc:
        emit("execution_uncertain", error=str(exc), instruction="Inspect MT5 before any retry")
        raise SystemExit(3) from None
    finally:
        try:
            if active is not None:
                tick = broker._fresh_tick_for_write()
                result = broker.close(active.ticket, 1.0, decision_id("session_end_close", tick.ts_ns, active.side))
                outcome = None
                for _ in range(12):
                    outcome = broker.closed_outcome(active.identifier)
                    if outcome is not None:
                        break
                    time.sleep(0.10)
                if outcome is not None:
                    realized_pnl += float(outcome["net_pnl"])
                    trades_closed += 1
                    emit("closed", reason="session_end", ticket=active.ticket,
                         fill_price=result.fill_price, net_pnl_usd=float(outcome["net_pnl"]),
                         session_net_pnl_usd=realized_pnl)
        finally:
            broker.shutdown()

    emit(
        "complete",
        demo_only=True,
        experimental=True,
        signals=signals,
        trades_opened=trades_opened,
        trades_closed=trades_closed,
        session_net_pnl_usd=realized_pnl,
    )


if __name__ == "__main__":
    main()
