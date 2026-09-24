from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from truetrade.scalper.adaptive_live import (
    AdaptiveLiveModel,
    FundamentalGate,
    LiveLabelQueue,
    TechnicalAnalyzer,
)
from truetrade.scalper.execution import DemoMT5Settings, ExecutionRejected, ExecutionUncertain
from truetrade.scalper.features import TickFeatureEngine
from truetrade.scalper.store import ScalperStore
from truetrade.scalper.timebase import BrokerTimebase, TimeNormalizedDemoMT5Execution
from truetrade.scalper.types import Side, Tick
from scripts.bbyg_v2_demo_autonomous import (
    ActiveTrade,
    close_trade,
    estimate_algorithmic_loss_usd,
    favorable_spreads,
    open_algorithmic_only_demo,
    update_algorithmic_protection,
)

FEATURE_WINDOW = 96
STRIDE = 4
MAX_GAP_NS = 300_000_000_000
MAX_ENTRY_DELAY_NS = 5_000_000_000


@dataclass
class PendingAdaptiveEntry:
    side: Side
    signal_ts_ns: int
    probability_long: float
    confidence: float
    signal_no: int
    copy_no: int
    technical_score: int
    regime: str


def emit(stage: str, **fields) -> None:
    print(json.dumps({"stage": stage, **fields}, sort_keys=True, default=str), flush=True)


def decision_id(kind: str, ts_ns: int, side: Side | None = None) -> str:
    raw = f"v3adaptive:{kind}:{ts_ns}:{'' if side is None else side.value}:{time.time_ns()}"
    return "v3_" + hashlib.sha256(raw.encode()).hexdigest()[:40]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive BBYG DEMO scalper: microstructure ML + technical confirmation + live champion/challenger learning."
    )
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=0.56)
    parser.add_argument("--exit-reversal-threshold", type=float, default=0.58)
    parser.add_argument("--min-exit-age-seconds", type=float, default=12.0)
    parser.add_argument("--model-reversal-confirmations", type=int, default=3)
    parser.add_argument("--technical-reversal-confirmations", type=int, default=3)
    parser.add_argument("--technical-reversal-score", type=int, default=4)
    parser.add_argument("--min-technical-score", type=int, default=3)
    parser.add_argument("--min-trend-efficiency", type=float, default=0.12)
    parser.add_argument("--min-velocity", type=float, default=0.05)
    parser.add_argument("--block-chop", action="store_true")
    parser.add_argument("--require-fundamental-feed", action="store_true")
    parser.add_argument("--fundamental-file", default=None)
    parser.add_argument("--fundamental-max-age-seconds", type=float, default=900.0)
    parser.add_argument("--entries-per-signal", type=int, default=3)
    parser.add_argument("--size", type=float, default=0.01)
    parser.add_argument("--max-trades", type=int, default=500)
    parser.add_argument("--max-open-positions", type=int, default=30)
    parser.add_argument("--max-same-side", type=int, default=30)
    parser.add_argument("--max-entries-per-second", type=int, default=8)
    parser.add_argument("--max-session-loss-usd", type=float, default=30.0)
    parser.add_argument("--max-aggregate-risk-usd", type=float, default=30.0)
    parser.add_argument("--algorithmic-loss-cut-spreads", type=float, default=3.0)
    parser.add_argument("--break-even-trigger-spreads", type=float, default=0.65)
    parser.add_argument("--profit-lock-trigger-spreads", type=float, default=0.85)
    parser.add_argument("--profit-lock-spreads", type=float, default=0.20)
    parser.add_argument("--trailing-trigger-spreads", type=float, default=1.00)
    parser.add_argument("--trailing-distance-spreads", type=float, default=0.45)
    parser.add_argument("--live-retrain-labels", type=int, default=120)
    parser.add_argument("--live-validation", type=int, default=120)
    parser.add_argument("--poll-ms", type=int, default=10)
    parser.add_argument("--status-seconds", type=float, default=10.0)
    args = parser.parse_args()

    if os.getenv("MT5_MODE", "demo").lower() != "demo":
        raise SystemExit("Refusing to run: MT5_MODE must be demo")
    if os.getenv("BBYG_DEMO_EXECUTION", "false").lower() != "true":
        raise SystemExit("Refusing to trade: BBYG_DEMO_EXECUTION must be true")
    if os.getenv("BBYG_AUTONOMOUS_DEMO_CONFIRM", "") != "I_UNDERSTAND_EXPERIMENTAL_DEMO":
        raise SystemExit("Set BBYG_AUTONOMOUS_DEMO_CONFIRM=I_UNDERSTAND_EXPERIMENTAL_DEMO")
    if os.getenv("BBYG_ALGO_ONLY_DEMO_CONFIRM", "") != "I_ACCEPT_NO_BROKER_STOP_DEMO_ONLY":
        raise SystemExit("Set BBYG_ALGO_ONLY_DEMO_CONFIRM=I_ACCEPT_NO_BROKER_STOP_DEMO_ONLY")
    if not 0.50 < args.threshold <= 0.80:
        raise SystemExit("--threshold must be in (0.50, 0.80]")
    if not 0.50 < args.exit_reversal_threshold <= 0.80:
        raise SystemExit("--exit-reversal-threshold must be in (0.50, 0.80]")
    if not 0 <= args.min_exit_age_seconds <= 300:
        raise SystemExit("--min-exit-age-seconds must be 0..300")
    if not 1 <= args.model_reversal_confirmations <= 20:
        raise SystemExit("--model-reversal-confirmations must be 1..20")
    if not 1 <= args.technical_reversal_confirmations <= 20:
        raise SystemExit("--technical-reversal-confirmations must be 1..20")
    if not 1 <= args.technical_reversal_score <= 4:
        raise SystemExit("--technical-reversal-score must be 1..4")
    if not 1 <= args.min_technical_score <= 4:
        raise SystemExit("--min-technical-score must be 1..4")
    if not 1 <= args.entries_per_signal <= 10:
        raise SystemExit("--entries-per-signal must be 1..10")
    if not 1 <= args.max_open_positions <= 50:
        raise SystemExit("--max-open-positions must be 1..50")
    if not 1 <= args.max_same_side <= args.max_open_positions:
        raise SystemExit("--max-same-side must be 1..max-open-positions")
    if not 1 <= args.max_entries_per_second <= 20:
        raise SystemExit("--max-entries-per-second must be 1..20")
    if not 0 < args.size <= 0.01:
        raise SystemExit("--size must be in (0, 0.01]")

    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-multiday-v1"))
    store = ScalperStore(state_dir / "scalper.sqlite")
    adaptive_dir = state_dir / "adaptive-v3"
    adaptive = AdaptiveLiveModel(
        store,
        adaptive_dir,
        live_validation=args.live_validation,
        retrain_every_labels=args.live_retrain_labels,
    )
    labels = LiveLabelQueue()
    technical = TechnicalAnalyzer()
    fundamental_path = Path(args.fundamental_file) if args.fundamental_file else adaptive_dir / "fundamental_context.json"
    fundamental = FundamentalGate(fundamental_path, max_age_seconds=args.fundamental_max_age_seconds)

    broker = TimeNormalizedDemoMT5Execution(DemoMT5Settings.from_env(), timebase=BrokerTimebase.from_env())
    features = TickFeatureEngine(window=FEATURE_WINDOW, min_ticks=FEATURE_WINDOW, fast_ticks=8, slow_ticks=24)
    pending: deque[PendingAdaptiveEntry] = deque()
    active: dict[str, ActiveTrade] = {}
    entry_times: deque[int] = deque()
    model_reversal_streak: dict[int, int] = {}
    technical_reversal_streak: dict[int, int] = {}

    tick_no = 0
    segment_tick_no = 0
    anchors = 0
    signals = 0
    blocked_ml = 0
    blocked_technical = 0
    blocked_fundamental = 0
    blocked_regime = 0
    blocked_microstructure = 0
    trades_opened = 0
    trades_closed = 0
    realized_pnl = 0.0
    last_tick: Tick | None = None
    last_p_long: float | None = None
    last_confidence: float | None = None
    last_technical = None
    last_fundamental = None
    last_learning = None
    last_status = 0.0
    deadline = time.monotonic() + args.hours * 3600.0
    uncertain = False

    try:
        broker.connect()
        account = broker._account(trading=True)
        start_equity = float(account.equity)
        if broker.positions():
            raise SystemExit("Refusing to start: existing BBYG position detected")
        emit(
            "started",
            demo_only=True,
            adaptive_v3=True,
            algorithmic_exits=True,
            broker_stop_role="none",
            model_generation=adaptive.generation,
            model_status=adaptive.status(),
            technical_confirmation=True,
            fundamental_feed=str(fundamental_path),
            require_fundamental_feed=args.require_fundamental_feed,
            threshold=args.threshold,
            min_technical_score=args.min_technical_score,
            entries_per_signal=args.entries_per_signal,
            min_exit_age_seconds=args.min_exit_age_seconds,
            model_reversal_confirmations=args.model_reversal_confirmations,
            technical_reversal_confirmations=args.technical_reversal_confirmations,
            technical_reversal_score=args.technical_reversal_score,
            start_equity=start_equity,
            symbol=broker.symbol,
        )

        while time.monotonic() < deadline:
            equity = broker.account_equity()
            if equity <= start_equity - args.max_session_loss_usd:
                emit("session_stop", reason="max_equity_drawdown", equity=equity, start_equity=start_equity)
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
                features = TickFeatureEngine(window=FEATURE_WINDOW, min_ticks=FEATURE_WINDOW, fast_ticks=8, slow_ticks=24)
                technical = TechnicalAnalyzer()
                segment_tick_no = 0
                emit("market_gap_reset", open_positions=len(active))

            tick_no += 1
            segment_tick_no += 1
            last_technical = technical.update(tick)

            resolved = labels.advance(tick)
            if resolved:
                adaptive.add_live_labels(resolved)
                emit("live_labels_resolved", count=len(resolved), total_live_labels=adaptive.status()["live_labels"])
            learning_result = adaptive.maybe_retrain()
            if learning_result is not None:
                last_learning = learning_result
                emit("live_learning_cycle", **learning_result, model_generation=adaptive.generation)

            # Reconcile externally/manual closed tickets.
            for ticket in list(active):
                trade = active.get(ticket)
                if trade is None:
                    continue
                current_ticket = broker.position_ticket_by_identifier(trade.identifier)
                if current_ticket is None:
                    outcome = wait_closed_outcome(broker, trade.identifier)
                    if outcome is None:
                        raise ExecutionUncertain(f"position vanished without outcome: {ticket}")
                    active.pop(ticket, None)
                    model_reversal_streak.pop(trade.identifier, None)
                    technical_reversal_streak.pop(trade.identifier, None)
                    pnl = float(outcome["net_pnl"])
                    realized_pnl += pnl
                    trades_closed += 1
                    emit("externally_closed", ticket=ticket, net_pnl_usd=pnl)
                elif current_ticket != ticket:
                    active.pop(ticket, None)
                    trade.ticket = current_ticket
                    active[current_ticket] = trade

            f = features.update(tick)
            new_score = f is not None and (segment_tick_no - FEATURE_WINDOW) % STRIDE == 0
            if new_score:
                anchors += 1
                labels.add_decision(f)
                last_p_long = adaptive.probability(f)
                last_confidence = max(last_p_long, 1.0 - last_p_long)
                last_fundamental = fundamental.snapshot()

            # Exit policy:
            # 1) hard software loss cut and already-earned profit protection may act immediately;
            # 2) model/technical reversals need both a minimum position age and persistent confirmation.
            # This prevents one noisy 4-tick score from liquidating an entire signal fanout.
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
                age_seconds = max(0.0, (tick.ts_ns - trade.opened_ns) / 1_000_000_000.0)
                reason = None
                exit_context: dict[str, object] = {}

                if trade.algo_floor_spreads is not None and fav <= trade.algo_floor_spreads:
                    reason = "algorithmic_profit_protection"
                    exit_context = {
                        "protection_stage": trade.protection_stage,
                        "floor_spreads": trade.algo_floor_spreads,
                        "peak_favorable_spreads": trade.peak_favorable_spreads,
                    }
                elif fav <= -args.algorithmic_loss_cut_spreads:
                    reason = "algorithmic_loss_cut"
                    exit_context = {"loss_cut_spreads": args.algorithmic_loss_cut_spreads}
                elif new_score and last_p_long is not None:
                    directional_p = last_p_long if trade.side is Side.LONG else 1.0 - last_p_long
                    opposite_p = 1.0 - directional_p
                    tech_same = 0
                    tech_opposite = 0
                    tech_ready = bool(last_technical and last_technical.ready)
                    if tech_ready:
                        tech_same = last_technical.score_for(trade.side)
                        tech_opposite = last_technical.opposite_score(trade.side)

                    model_reverse_now = (
                        opposite_p >= args.exit_reversal_threshold
                        and (not tech_ready or tech_opposite >= tech_same)
                    )
                    technical_reverse_now = (
                        tech_ready
                        and tech_opposite >= args.technical_reversal_score
                        and tech_opposite > tech_same
                        and directional_p < 0.50
                    )

                    if model_reverse_now:
                        model_reversal_streak[trade.identifier] = model_reversal_streak.get(trade.identifier, 0) + 1
                    else:
                        model_reversal_streak[trade.identifier] = 0
                    if technical_reverse_now:
                        technical_reversal_streak[trade.identifier] = technical_reversal_streak.get(trade.identifier, 0) + 1
                    else:
                        technical_reversal_streak[trade.identifier] = 0

                    if age_seconds >= args.min_exit_age_seconds:
                        if model_reversal_streak[trade.identifier] >= args.model_reversal_confirmations:
                            reason = "confirmed_adaptive_model_reversal"
                        elif technical_reversal_streak[trade.identifier] >= args.technical_reversal_confirmations:
                            reason = "confirmed_technical_reversal"

                    exit_context = {
                        "directional_probability": directional_p,
                        "opposite_probability": opposite_p,
                        "technical_same_score": tech_same,
                        "technical_opposite_score": tech_opposite,
                        "model_reversal_streak": model_reversal_streak.get(trade.identifier, 0),
                        "technical_reversal_streak": technical_reversal_streak.get(trade.identifier, 0),
                    }

                if reason is not None:
                    emit(
                        "exit_decision",
                        ticket=trade.ticket,
                        side=trade.side.value,
                        reason=reason,
                        age_seconds=age_seconds,
                        favorable_spreads=fav,
                        **exit_context,
                    )
                    pnl = close_trade(broker, trade, tick, reason)
                    active.pop(ticket, None)
                    active.pop(trade.ticket, None)
                    model_reversal_streak.pop(trade.identifier, None)
                    technical_reversal_streak.pop(trade.identifier, None)
                    realized_pnl += pnl
                    trades_closed += 1

            # New entries require ML + microstructure + technical confirmation + optional fundamentals.
            if new_score and f is not None and last_p_long is not None and last_confidence is not None:
                if last_confidence < args.threshold:
                    blocked_ml += 1
                else:
                    side = Side.LONG if last_p_long >= 0.5 else Side.SHORT
                    sign = side.sign
                    micro_ok = (
                        f.trend_efficiency * sign >= args.min_trend_efficiency
                        and f.fast_velocity * sign >= args.min_velocity
                    )
                    tech_ok = bool(
                        last_technical
                        and last_technical.ready
                        and last_technical.score_for(side) >= args.min_technical_score
                        and last_technical.score_for(side) > last_technical.opposite_score(side)
                    )
                    regime_ok = not (args.block_chop and last_technical and last_technical.regime == "chop")
                    fund = last_fundamental or fundamental.snapshot()
                    fund_ok = fundamental.allows(fund, side, require_active=args.require_fundamental_feed)
                    if not micro_ok:
                        blocked_microstructure += 1
                    elif not tech_ok:
                        blocked_technical += 1
                    elif not regime_ok:
                        blocked_regime += 1
                    elif not fund_ok:
                        blocked_fundamental += 1
                    else:
                        signals += 1
                        for copy_no in range(1, args.entries_per_signal + 1):
                            pending.append(PendingAdaptiveEntry(
                                side, tick.ts_ns, last_p_long, last_confidence,
                                signals, copy_no, last_technical.score_for(side), last_technical.regime,
                            ))
                        emit(
                            "qualified_signal",
                            signal_no=signals,
                            side=side.value,
                            p_long=last_p_long,
                            confidence=last_confidence,
                            technical_score=last_technical.score_for(side),
                            opposite_technical_score=last_technical.opposite_score(side),
                            rsi=last_technical.rsi,
                            ema_fast=last_technical.ema_fast,
                            ema_slow=last_technical.ema_slow,
                            atr_spreads=last_technical.atr_spreads,
                            regime=last_technical.regime,
                            trend_efficiency=f.trend_efficiency,
                            fast_velocity=f.fast_velocity,
                            model_generation=adaptive.generation,
                            fundamental_mode=fund.mode,
                            fundamental_active=fund.active,
                            fanout=args.entries_per_signal,
                        )

            while entry_times and tick.ts_ns - entry_times[0] >= 1_000_000_000:
                entry_times.popleft()
            carry: deque[PendingAdaptiveEntry] = deque()
            for _ in range(len(pending)):
                item = pending.popleft()
                delay_ns = tick.ts_ns - item.signal_ts_ns
                if delay_ns <= 0:
                    carry.append(item)
                    continue
                if delay_ns > MAX_ENTRY_DELAY_NS or trades_opened >= args.max_trades:
                    continue
                if len(active) >= args.max_open_positions:
                    carry.append(item)
                    continue
                if sum(1 for t in active.values() if t.side is item.side) >= args.max_same_side:
                    carry.append(item)
                    continue
                if len(entry_times) >= args.max_entries_per_second:
                    carry.append(item)
                    continue

                info = broker._symbol_info()
                volume = broker._normalized_volume(args.size, info)
                fresh = broker._fresh_tick_for_write()
                expected = fresh.ask if item.side is Side.LONG else fresh.bid
                candidate_risk = estimate_algorithmic_loss_usd(
                    broker, item.side, volume, expected, max(fresh.spread, 1e-12), args.algorithmic_loss_cut_spreads
                )
                current_risk = sum(float(t.risk_budget_usd) for t in active.values())
                if current_risk + candidate_risk > args.max_aggregate_risk_usd + 1e-9:
                    carry.append(item)
                    continue

                opened, software_risk, observed_spread = open_algorithmic_only_demo(
                    broker,
                    item.side,
                    args.size,
                    decision_id("open", tick.ts_ns, item.side),
                    loss_cut_spreads=args.algorithmic_loss_cut_spreads,
                )
                positions = broker.positions()
                pos = next((p for p in positions if p.position_id == opened.position_id), None)
                if opened.position_id is None or opened.position_identifier is None or pos is None:
                    raise ExecutionUncertain("opened adaptive position not observable")
                trade = ActiveTrade(
                    ticket=opened.position_id,
                    identifier=int(opened.position_identifier),
                    side=item.side,
                    entry=float(pos.entry),
                    entry_spread=observed_spread,
                    entry_tick_no=tick_no,
                    opened_ns=tick.ts_ns,
                    risk_budget_usd=software_risk,
                    signal_no=item.signal_no,
                    copy_no=item.copy_no,
                )
                active[trade.ticket] = trade
                model_reversal_streak[trade.identifier] = 0
                technical_reversal_streak[trade.identifier] = 0
                trades_opened += 1
                entry_times.append(tick.ts_ns)
                emit(
                    "opened",
                    ticket=trade.ticket,
                    side=item.side.value,
                    signal_no=item.signal_no,
                    copy_no=item.copy_no,
                    entry=trade.entry,
                    confidence=item.confidence,
                    technical_score=item.technical_score,
                    regime=item.regime,
                    model_generation=adaptive.generation,
                    broker_stop=None,
                    broker_tp=None,
                    open_positions=len(active),
                    trades_opened=trades_opened,
                )
            pending = carry

            now = time.monotonic()
            if now - last_status >= args.status_seconds:
                emit(
                    "status",
                    ticks=tick_no,
                    anchors=anchors,
                    signals=signals,
                    p_long=last_p_long,
                    confidence=last_confidence,
                    model_generation=adaptive.generation,
                    model_qualified=adaptive.qualified,
                    live_labels=adaptive.status()["live_labels"],
                    pending_live_labels=len(labels.pending),
                    learning=last_learning,
                    technical_ready=None if last_technical is None else last_technical.ready,
                    technical_long_score=None if last_technical is None else last_technical.long_score,
                    technical_short_score=None if last_technical is None else last_technical.short_score,
                    regime=None if last_technical is None else last_technical.regime,
                    fundamental_active=None if last_fundamental is None else last_fundamental.active,
                    fundamental_mode=None if last_fundamental is None else last_fundamental.mode,
                    blocked_ml=blocked_ml,
                    blocked_microstructure=blocked_microstructure,
                    blocked_technical=blocked_technical,
                    blocked_regime=blocked_regime,
                    blocked_fundamental=blocked_fundamental,
                    queued_entries=len(pending),
                    open_positions=len(active),
                    trades_opened=trades_opened,
                    trades_closed=trades_closed,
                    realized_pnl_usd=realized_pnl,
                    equity=broker.account_equity(),
                    max_model_reversal_streak=max(model_reversal_streak.values(), default=0),
                    max_technical_reversal_streak=max(technical_reversal_streak.values(), default=0),
                    exit_grace_positions=sum(
                        1 for t in active.values()
                        if (tick.ts_ns - t.opened_ns) / 1_000_000_000.0 < args.min_exit_age_seconds
                    ),
                )
                last_status = now

            last_tick = tick
            time.sleep(args.poll_ms / 1000.0)

    except ExecutionRejected as exc:
        emit("execution_rejected", error=str(exc))
        raise SystemExit(2) from None
    except ExecutionUncertain as exc:
        uncertain = True
        emit("execution_uncertain", error=str(exc), instruction="Inspect MT5 before retry")
        raise SystemExit(3) from None
    finally:
        if broker.connected and not uncertain:
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
                        model_reversal_streak.pop(trade.identifier, None)
                        technical_reversal_streak.pop(trade.identifier, None)
                        realized_pnl += pnl
                        trades_closed += 1
                    except Exception as exc:
                        emit("final_flatten_failed", ticket=ticket, error=str(exc))
            finally:
                broker.shutdown()
        elif broker.connected:
            broker.shutdown()
        store.close()

    emit(
        "complete",
        adaptive_v3=True,
        model_generation=adaptive.generation,
        live_labels=adaptive.status()["live_labels"],
        signals=signals,
        trades_opened=trades_opened,
        trades_closed=trades_closed,
        session_net_pnl_usd=realized_pnl,
        uncertain=uncertain,
    )


if __name__ == "__main__":
    main()
