from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from truetrade.scalper.execution import DemoMT5Settings
from truetrade.scalper.features import TickFeatureEngine
from truetrade.scalper.research_models import RobustScaler, fit_logit
from truetrade.scalper.store import ScalperStore
from truetrade.scalper.types import Tick

THRESHOLDS = (0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60)
MAX_POSITIONS = 6
MAX_SAME_SIDE = 4
MAX_ENTRIES_PER_SECOND = 4
MAX_ENTRY_DELAY_NS = 5_000_000_000
MAX_GAP_NS = 300_000_000_000
MAX_HOLD_TICKS = 600
TARGET_SPREADS = 1.8
STOP_SPREADS = 1.4
COST_SPREADS = 0.20
STRIDE = 4
FEATURE_WINDOW = 96


@dataclass
class PendingSignal:
    signal_ts_ns: int
    p_long: float
    confidence: float


@dataclass
class ShadowPosition:
    side: int
    signal_ts_ns: int
    entry_ts_ns: int
    entry_tick_no: int
    entry: float
    entry_spread: float
    target: float
    stop: float


@dataclass
class ShadowState:
    threshold: float
    pending: list[PendingSignal] = field(default_factory=list)
    positions: list[ShadowPosition] = field(default_factory=list)
    entry_times: deque[int] = field(default_factory=deque)
    signals_selected: int = 0
    entries_opened: int = 0
    blocked_position_limit: int = 0
    blocked_same_side_limit: int = 0
    blocked_rate_limit: int = 0
    entry_too_late: int = 0
    target_exits: int = 0
    stop_exits: int = 0
    horizon_exits: int = 0
    end_exits: int = 0
    long_trades: int = 0
    short_trades: int = 0
    gross_pnl_spreads: float = 0.0
    net_pnl_spreads: float = 0.0
    closed_trades: int = 0
    wins: int = 0
    gross_gains_net_cost: float = 0.0
    gross_losses_net_cost: float = 0.0
    peak_curve: float = 0.0
    max_drawdown_spreads: float = 0.0

    def _record_close(self, position: ShadowPosition, exit_price: float, reason: str) -> dict:
        gross = ((float(exit_price) - position.entry) * position.side) / max(position.entry_spread, 1e-12)
        net = float(gross - COST_SPREADS)
        self.gross_pnl_spreads += float(gross)
        self.net_pnl_spreads += net
        self.closed_trades += 1
        if net > 0:
            self.wins += 1
            self.gross_gains_net_cost += net
        elif net < 0:
            self.gross_losses_net_cost += -net
        self.peak_curve = max(self.peak_curve, self.net_pnl_spreads)
        self.max_drawdown_spreads = max(self.max_drawdown_spreads, self.peak_curve - self.net_pnl_spreads)
        if reason == "target":
            self.target_exits += 1
        elif reason == "stop":
            self.stop_exits += 1
        elif reason == "horizon":
            self.horizon_exits += 1
        elif reason == "end":
            self.end_exits += 1
        return {
            "threshold": self.threshold,
            "reason": reason,
            "side": "LONG" if position.side > 0 else "SHORT",
            "signal_ts_ns": position.signal_ts_ns,
            "entry_ts_ns": position.entry_ts_ns,
            "entry": position.entry,
            "exit": float(exit_price),
            "gross_pnl_spreads": float(gross),
            "net_pnl_spreads": net,
        }

    def process_tick(self, tick: Tick, tick_no: int) -> list[dict]:
        events: list[dict] = []
        survivors: list[ShadowPosition] = []
        for position in self.positions:
            exit_price = None
            reason = None
            if position.side > 0:
                if tick.bid <= position.stop:
                    exit_price, reason = tick.bid, "stop"
                elif tick.bid >= position.target:
                    exit_price, reason = position.target, "target"
            else:
                if tick.ask >= position.stop:
                    exit_price, reason = tick.ask, "stop"
                elif tick.ask <= position.target:
                    exit_price, reason = position.target, "target"
            if reason is None and tick_no - position.entry_tick_no >= MAX_HOLD_TICKS:
                exit_price = tick.bid if position.side > 0 else tick.ask
                reason = "horizon"
            if reason is not None:
                events.append({"event": "shadow_close", **self._record_close(position, float(exit_price), reason)})
            else:
                survivors.append(position)
        self.positions = survivors

        pending = self.pending
        self.pending = []
        for signal in pending:
            delay = tick.ts_ns - signal.signal_ts_ns
            if delay <= 0:
                self.pending.append(signal)
                continue
            if delay > MAX_ENTRY_DELAY_NS:
                self.entry_too_late += 1
                continue
            side = 1 if signal.p_long >= 0.5 else -1
            while self.entry_times and tick.ts_ns - self.entry_times[0] >= 1_000_000_000:
                self.entry_times.popleft()
            if len(self.entry_times) >= MAX_ENTRIES_PER_SECOND:
                self.blocked_rate_limit += 1
                continue
            if len(self.positions) >= MAX_POSITIONS:
                self.blocked_position_limit += 1
                continue
            if sum(1 for p in self.positions if p.side == side) >= MAX_SAME_SIDE:
                self.blocked_same_side_limit += 1
                continue
            spread = max(tick.spread, 1e-12)
            if side > 0:
                entry = tick.ask
                target = entry + TARGET_SPREADS * spread
                stop = entry - STOP_SPREADS * spread
                self.long_trades += 1
            else:
                entry = tick.bid
                target = entry - TARGET_SPREADS * spread
                stop = entry + STOP_SPREADS * spread
                self.short_trades += 1
            position = ShadowPosition(
                side=side,
                signal_ts_ns=signal.signal_ts_ns,
                entry_ts_ns=tick.ts_ns,
                entry_tick_no=tick_no,
                entry=float(entry),
                entry_spread=float(spread),
                target=float(target),
                stop=float(stop),
            )
            self.positions.append(position)
            self.entry_times.append(tick.ts_ns)
            self.entries_opened += 1
            events.append({
                "event": "shadow_open",
                "threshold": self.threshold,
                "side": "LONG" if side > 0 else "SHORT",
                "signal_ts_ns": signal.signal_ts_ns,
                "entry_ts_ns": tick.ts_ns,
                "p_long": signal.p_long,
                "confidence": signal.confidence,
                "entry": float(entry),
                "target": float(target),
                "stop": float(stop),
                "spread": float(spread),
            })
        return events

    def schedule(self, tick: Tick, p_long: float) -> None:
        confidence = max(float(p_long), 1.0 - float(p_long))
        if confidence >= self.threshold:
            self.signals_selected += 1
            self.pending.append(PendingSignal(tick.ts_ns, float(p_long), float(confidence)))

    def flatten(self, tick: Tick) -> list[dict]:
        events = []
        for position in self.positions:
            exit_price = tick.bid if position.side > 0 else tick.ask
            events.append({"event": "shadow_close", **self._record_close(position, float(exit_price), "end")})
        self.positions = []
        self.pending = []
        return events

    def summary(self) -> dict:
        pf = None if self.gross_losses_net_cost <= 1e-12 else self.gross_gains_net_cost / self.gross_losses_net_cost
        win_rate = None if self.closed_trades == 0 else self.wins / self.closed_trades
        return {
            "threshold": self.threshold,
            "signals_selected": self.signals_selected,
            "entries_opened": self.entries_opened,
            "closed_trades": self.closed_trades,
            "open_positions": len(self.positions),
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "target_exits": self.target_exits,
            "stop_exits": self.stop_exits,
            "horizon_exits": self.horizon_exits,
            "end_exits": self.end_exits,
            "entry_too_late": self.entry_too_late,
            "blocked_position_limit": self.blocked_position_limit,
            "blocked_same_side_limit": self.blocked_same_side_limit,
            "blocked_rate_limit": self.blocked_rate_limit,
            "gross_pnl_spreads": self.gross_pnl_spreads,
            "net_pnl_spreads": self.net_pnl_spreads,
            "profit_factor": pf,
            "win_rate": win_rate,
            "max_drawdown_spreads": self.max_drawdown_spreads,
        }


def _load_causal_training(store: ScalperStore, cutoff_ns: int) -> tuple[np.ndarray, np.ndarray, int]:
    rows = list(store.db.execute(
        """SELECT s.x_json,s.y,i.label_end_ts_ns
           FROM samples s JOIN sample_label_intervals i ON i.feature_ts_ns=s.feature_ts_ns
           WHERE i.label_end_ts_ns < ? ORDER BY s.id DESC LIMIT 20000""",
        (int(cutoff_ns),),
    ))
    rows.reverse()
    if len(rows) < 10_000:
        raise SystemExit(f"insufficient causal training labels before shadow start: {len(rows)}")
    x = np.asarray([json.loads(r[0]) for r in rows], dtype=float)
    y = np.asarray([int(r[1]) for r in rows], dtype=np.int8)
    if x.shape != (len(rows), 8) or not np.isfinite(x).all():
        raise SystemExit("invalid causal training matrix")
    return x, y, int(max(int(r[2]) for r in rows))


def _resolve_symbol(mt5, requested: str) -> str:
    symbols = mt5.symbols_get()
    if symbols is None:
        raise SystemExit(f"MT5 symbols_get failed: {mt5.last_error()}")
    names = {x.name for x in symbols}
    if requested in names:
        return requested
    matches = sorted(x for x in names if x.startswith(requested))
    if len(matches) != 1:
        raise SystemExit(f"symbol missing or ambiguous: {requested}")
    return matches[0]


def _write_event(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only BBYG v2 live shadow runner; NEVER sends MT5 orders")
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--poll-ms", type=int, default=10)
    parser.add_argument("--status-seconds", type=float, default=30.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if not 0.05 <= args.hours <= 12:
        raise SystemExit("--hours must be between 0.05 and 12")
    if not 5 <= args.poll_ms <= 1000:
        raise SystemExit("--poll-ms must be between 5 and 1000")
    if os.getenv("MT5_MODE", "demo").lower() != "demo":
        raise SystemExit("shadow runner requires MT5_MODE=demo")
    if os.getenv("BBYG_DEMO_EXECUTION", "false").lower() != "false":
        raise SystemExit("refusing shadow mode while BBYG_DEMO_EXECUTION is enabled")

    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise SystemExit("MetaTrader5 package is required on Windows") from None

    settings = DemoMT5Settings.from_env()
    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-multiday-v1"))
    store = ScalperStore(state_dir / "scalper.sqlite")
    start_wall_ns = time.time_ns()
    train_x, train_y, latest_train_label_end_ns = _load_causal_training(store, start_wall_ns)
    store.close()

    scaler = RobustScaler.fit(train_x)
    model = fit_logit(scaler.transform(train_x), train_y, iterations=140, balanced=True)

    if not mt5.initialize(settings.terminal_path, login=settings.login, password=settings.password,
                          server=settings.server, timeout=15000):
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        account = mt5.account_info()
        terminal = mt5.terminal_info()
        if account is None or terminal is None or not terminal.connected:
            raise SystemExit(f"MT5 account/terminal unavailable: {mt5.last_error()}")
        if int(account.login) != int(settings.login) or str(account.server) != str(settings.server):
            raise SystemExit("MT5 account identity mismatch")
        if account.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
            raise SystemExit("shadow runner refuses non-DEMO accounts")

        symbol = _resolve_symbol(mt5, settings.symbol)
        info = mt5.symbol_info(symbol)
        if info is None:
            raise SystemExit("MT5 symbol_info unavailable")
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise SystemExit("MT5 symbol selection failed")

        out_path = Path(args.output) if args.output else state_dir / (
            "shadow_v2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".jsonl"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        duration_seconds = args.hours * 3600.0
        deadline = time.monotonic() + duration_seconds
        feature_engine = TickFeatureEngine(window=FEATURE_WINDOW, min_ticks=FEATURE_WINDOW,
                                           fast_ticks=8, slow_ticks=24)
        states = {t: ShadowState(t) for t in THRESHOLDS}
        last_signature = None
        last_tick: Tick | None = None
        tick_no = 0
        segment_tick_no = 0
        anchors_evaluated = 0
        last_status = 0.0
        offset_ns = int(os.getenv("MT5_SERVER_UTC_OFFSET_SECONDS", "0")) * 1_000_000_000

        with out_path.open("a", encoding="utf-8") as handle:
            start_payload = {
                "event": "shadow_start",
                "read_only": True,
                "orders_sent": False,
                "mode": "v2_all_anchor_live_shadow",
                "login": int(account.login),
                "server": str(account.server),
                "symbol": symbol,
                "duration_hours": args.hours,
                "thresholds": THRESHOLDS,
                "training_samples": int(len(train_x)),
                "training_long_fraction": float(np.mean(train_y)),
                "latest_training_label_end_ns": latest_train_label_end_ns,
                "stride": STRIDE,
                "feature_window": FEATURE_WINDOW,
                "target_spreads": TARGET_SPREADS,
                "stop_spreads": STOP_SPREADS,
                "cost_spreads": COST_SPREADS,
            }
            _write_event(handle, start_payload)
            print(json.dumps({**start_payload, "output": str(out_path)}, sort_keys=True), flush=True)

            while time.monotonic() < deadline:
                raw = mt5.symbol_info_tick(symbol)
                if raw is None:
                    time.sleep(args.poll_ms / 1000.0)
                    continue
                signature = (int(raw.time_msc), float(raw.bid), float(raw.ask),
                             float(getattr(raw, "last", 0.0) or 0.0),
                             float(getattr(raw, "volume_real", getattr(raw, "volume", 0.0)) or 0.0))
                if signature == last_signature:
                    time.sleep(args.poll_ms / 1000.0)
                    continue
                last_signature = signature
                ts_ns = int(raw.time_msc) * 1_000_000 - offset_ns
                if last_tick is not None and ts_ns <= last_tick.ts_ns:
                    ts_ns = last_tick.ts_ns + 1
                tick = Tick(ts_ns, float(raw.bid), float(raw.ask), signature[3], signature[4])

                if last_tick is not None and tick.ts_ns - last_tick.ts_ns > MAX_GAP_NS:
                    for state in states.values():
                        for event in state.flatten(last_tick):
                            event["reason_override"] = "market_gap"
                            _write_event(handle, event)
                    feature_engine = TickFeatureEngine(window=FEATURE_WINDOW, min_ticks=FEATURE_WINDOW,
                                                       fast_ticks=8, slow_ticks=24)
                    segment_tick_no = 0
                    _write_event(handle, {"event": "market_gap_reset", "gap_ns": tick.ts_ns - last_tick.ts_ns})

                tick_no += 1
                segment_tick_no += 1
                for state in states.values():
                    for event in state.process_tick(tick, tick_no):
                        _write_event(handle, event)

                features = feature_engine.update(tick)
                # Start an all-anchor cadence once the 96-tick causal warmup is complete.
                if features is not None and (segment_tick_no - FEATURE_WINDOW) % STRIDE == 0:
                    vector = np.asarray([features.vector()], dtype=float)
                    p_long = float(model.probability(scaler.transform(vector))[0])
                    confidence = max(p_long, 1.0 - p_long)
                    anchors_evaluated += 1
                    _write_event(handle, {
                        "event": "shadow_signal",
                        "tick_ts_ns": tick.ts_ns,
                        "p_long": p_long,
                        "confidence": confidence,
                        "mid": tick.mid,
                        "spread": tick.spread,
                    })
                    for state in states.values():
                        state.schedule(tick, p_long)

                last_tick = tick
                now = time.monotonic()
                if now - last_status >= args.status_seconds:
                    status = {
                        "event": "shadow_status",
                        "elapsed_seconds": duration_seconds - max(0.0, deadline - now),
                        "ticks": tick_no,
                        "anchors_evaluated": anchors_evaluated,
                        "last_tick_ts_ns": tick.ts_ns,
                        "thresholds": [states[t].summary() for t in THRESHOLDS],
                    }
                    _write_event(handle, status)
                    print(json.dumps(status, sort_keys=True), flush=True)
                    last_status = now
                time.sleep(args.poll_ms / 1000.0)

            if last_tick is not None:
                for state in states.values():
                    for event in state.flatten(last_tick):
                        _write_event(handle, event)
            final = {
                "event": "shadow_complete",
                "read_only": True,
                "orders_sent": False,
                "ticks": tick_no,
                "anchors_evaluated": anchors_evaluated,
                "thresholds": [states[t].summary() for t in THRESHOLDS],
                "output": str(out_path),
            }
            _write_event(handle, final)
            print(json.dumps(final, sort_keys=True), flush=True)
    except KeyboardInterrupt:
        print(json.dumps({"event": "shadow_interrupted", "read_only": True, "orders_sent": False}), flush=True)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
