from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .types import MicroFeatures, Tick


FRAME_SECONDS = (1, 5, 60)
FRAME_FEATURES = (
    "return",
    "ema9_distance",
    "ema21_distance",
    "ema_cross",
    "ema9_slope",
    "rsi14",
    "atr14",
    "realized_volatility",
    "range_position",
    "breakout_distance",
    "relative_volume",
)
MICRO_FEATURE_NAMES = (
    "spread_z",
    "fast_velocity",
    "slow_velocity",
    "acceleration",
    "tick_imbalance",
    "trend_efficiency",
    "volatility_ratio",
    "last_move_ratio",
)


@dataclass(frozen=True)
class ExecutionContext:
    """Causal execution-quality context observable at decision time."""

    latency_ms: float = 0.0
    slippage_spreads: float = 0.0
    recent_failure_rate: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (
            self.latency_ms, self.slippage_spreads, self.recent_failure_rate
        )):
            raise ValueError("execution context must be finite")
        if self.latency_ms < 0 or not 0.0 <= self.recent_failure_rate <= 1.0:
            raise ValueError("invalid execution context")

    def vector(self) -> tuple[float, float, float]:
        return (
            math.tanh(self.latency_ms / 500.0),
            math.tanh(self.slippage_spreads / 2.0),
            float(self.recent_failure_rate),
        )


@dataclass(frozen=True)
class MarketState:
    ts_ns: int
    embedding: tuple[float, ...]
    regime: str
    feature_names: tuple[str, ...]
    context: dict

    def __post_init__(self) -> None:
        if self.ts_ns <= 0:
            raise ValueError("market-state timestamp must be positive")
        if len(self.embedding) != len(self.feature_names) or not self.embedding:
            raise ValueError("market-state feature schema mismatch")
        if not all(math.isfinite(float(v)) for v in self.embedding):
            raise ValueError("market-state embedding must be finite")
        if self.regime not in {"quiet", "range", "trend_up", "trend_down", "shock"}:
            raise ValueError("unknown market regime")


@dataclass
class _Bar:
    bucket: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class _BarFrame:
    def __init__(self, seconds: int, *, max_bars: int = 256):
        if seconds <= 0:
            raise ValueError("bar seconds must be positive")
        self.seconds = int(seconds)
        self.completed: deque[_Bar] = deque(maxlen=max_bars)
        self.current: _Bar | None = None

    def update(self, tick: Tick) -> None:
        bucket = tick.ts_ns // (self.seconds * 1_000_000_000)
        price = tick.mid
        if self.current is None:
            self.current = _Bar(bucket, price, price, price, price, float(tick.volume))
            return
        if bucket < self.current.bucket:
            raise ValueError("ticks must be chronological")
        if bucket == self.current.bucket:
            self.current.high = max(self.current.high, price)
            self.current.low = min(self.current.low, price)
            self.current.close = price
            self.current.volume += float(tick.volume)
            return
        self.completed.append(self.current)
        self.current = _Bar(bucket, price, price, price, price, float(tick.volume))

    def bars(self) -> list[_Bar]:
        rows = list(self.completed)
        if self.current is not None:
            rows.append(self.current)
        return rows


@dataclass(frozen=True)
class _FrameSnapshot:
    vector: tuple[float, ...]
    atr_price: float
    atr_spreads: float
    ema_cross_price: float
    trend_strength: float
    range_position: float
    shock_ratio: float


def _ema(values: np.ndarray, period: int) -> float:
    alpha = 2.0 / (period + 1.0)
    result = float(values[0])
    for value in values[1:]:
        result = alpha * float(value) + (1.0 - alpha) * result
    return result


def _rsi(values: np.ndarray, period: int = 14) -> float:
    if len(values) < 2:
        return 0.0
    diffs = np.diff(values[-(period + 1):])
    if len(diffs) == 0:
        return 0.0
    gains = np.maximum(diffs, 0.0)
    losses = np.maximum(-diffs, 0.0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_gain + avg_loss <= 1e-12:
        return 0.0
    raw = avg_gain / (avg_gain + avg_loss)
    return float(np.clip(raw * 2.0 - 1.0, -1.0, 1.0))


def _true_ranges(bars: list[_Bar]) -> np.ndarray:
    if not bars:
        return np.empty(0, dtype=float)
    result = np.empty(len(bars), dtype=float)
    result[0] = bars[0].high - bars[0].low
    for i in range(1, len(bars)):
        prev_close = bars[i - 1].close
        result[i] = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - prev_close),
            abs(bars[i].low - prev_close),
        )
    return result


def _frame_snapshot(bars: list[_Bar], spread: float) -> _FrameSnapshot:
    if len(bars) < 3:
        raise ValueError("insufficient bars")
    spread = max(float(spread), 1e-12)
    closes = np.asarray([b.close for b in bars], dtype=float)
    highs = np.asarray([b.high for b in bars], dtype=float)
    lows = np.asarray([b.low for b in bars], dtype=float)
    volumes = np.asarray([b.volume for b in bars], dtype=float)

    ema9 = _ema(closes[-64:], 9)
    ema21 = _ema(closes[-96:], 21)
    prior_ema9 = _ema(closes[-65:-1], 9) if len(closes) >= 4 else ema9
    ema_cross_price = ema9 - ema21

    trs = _true_ranges(bars)
    atr = float(np.mean(trs[-min(14, len(trs)):]))
    historical_tr = trs[-min(30, len(trs)):-1]
    tr_baseline = float(np.median(historical_tr)) if len(historical_tr) else max(atr, spread)
    shock_ratio = float(trs[-1] / max(tr_baseline, spread, 1e-12))

    log_returns = np.diff(np.log(np.maximum(closes, 1e-12)))
    recent_returns = log_returns[-min(20, len(log_returns)):]
    realized_price = (
        float(np.std(recent_returns) * closes[-1]) if len(recent_returns) >= 2 else 0.0
    )

    lookback = min(20, len(bars))
    local_high = float(np.max(highs[-lookback:]))
    local_low = float(np.min(lows[-lookback:]))
    local_range = max(local_high - local_low, spread)
    range_position = float(np.clip(
        2.0 * (closes[-1] - local_low) / local_range - 1.0, -1.0, 1.0
    ))

    prior_highs = highs[-min(21, len(highs)):-1]
    prior_lows = lows[-min(21, len(lows)):-1]
    breakout_spreads = 0.0
    if len(prior_highs):
        prior_high = float(np.max(prior_highs))
        prior_low = float(np.min(prior_lows))
        if closes[-1] > prior_high:
            breakout_spreads = (closes[-1] - prior_high) / spread
        elif closes[-1] < prior_low:
            breakout_spreads = (closes[-1] - prior_low) / spread

    previous_volumes = volumes[-min(21, len(volumes)):-1]
    volume_baseline = float(np.mean(previous_volumes)) if len(previous_volumes) else volumes[-1]
    relative_volume = (
        volumes[-1] / volume_baseline - 1.0 if volume_baseline > 1e-12 else 0.0
    )

    return_spreads = (closes[-1] - closes[-2]) / spread
    ema9_distance_spreads = (closes[-1] - ema9) / spread
    ema21_distance_spreads = (closes[-1] - ema21) / spread
    ema_cross_spreads = ema_cross_price / spread
    ema9_slope_spreads = (ema9 - prior_ema9) / spread
    atr_spreads = atr / spread
    rv_spreads = realized_price / spread
    trend_strength = abs(ema_cross_price) / max(atr, spread, 1e-12)

    vector = (
        math.tanh(return_spreads / 3.0),
        math.tanh(ema9_distance_spreads / 4.0),
        math.tanh(ema21_distance_spreads / 8.0),
        math.tanh(ema_cross_spreads / 6.0),
        math.tanh(ema9_slope_spreads / 2.0),
        _rsi(closes, 14),
        math.tanh(atr_spreads / 10.0),
        math.tanh(rv_spreads / 8.0),
        range_position,
        math.tanh(breakout_spreads / 4.0),
        math.tanh(relative_volume),
    )
    return _FrameSnapshot(
        vector=tuple(float(v) for v in vector),
        atr_price=atr,
        atr_spreads=atr_spreads,
        ema_cross_price=ema_cross_price,
        trend_strength=trend_strength,
        range_position=range_position,
        shock_ratio=shock_ratio,
    )


class MarketStateEncoder:
    """Causal multi-timescale market state used by BBYG v4.

    Replay and live code must both feed ticks sequentially through this class. No candle
    is synthesized for missing time buckets, and the current partial candle contains only
    information observable up to the current tick. `compute=False` advances all causal
    bar state without materializing an embedding; this is used by strided offline replay.
    """

    MIN_BARS = {1: 30, 5: 24, 60: 16}

    def __init__(self):
        self.frames = {seconds: _BarFrame(seconds) for seconds in FRAME_SECONDS}
        self.last_ts_ns = 0
        names = list(MICRO_FEATURE_NAMES)
        for seconds in FRAME_SECONDS:
            names.extend(f"{seconds}s_{name}" for name in FRAME_FEATURES)
        names.extend((
            "utc_time_sin",
            "utc_time_cos",
            "spread_fraction",
            "execution_latency",
            "execution_slippage",
            "execution_failure_rate",
        ))
        self.feature_names = tuple(names)

    @property
    def dimensions(self) -> int:
        return len(self.feature_names)

    def update(
        self,
        tick: Tick,
        micro: MicroFeatures | None,
        *,
        execution: ExecutionContext | None = None,
        compute: bool = True,
    ) -> MarketState | None:
        if self.last_ts_ns and tick.ts_ns <= self.last_ts_ns:
            raise ValueError("ticks must be strictly increasing")
        self.last_ts_ns = tick.ts_ns
        for frame in self.frames.values():
            frame.update(tick)
        if micro is None or not compute:
            return None

        snapshots: dict[int, _FrameSnapshot] = {}
        for seconds, frame in self.frames.items():
            bars = frame.bars()
            if len(bars) < self.MIN_BARS[seconds]:
                return None
            snapshots[seconds] = _frame_snapshot(bars, micro.spread)

        micro_vector = (
            micro.spread_z,
            micro.fast_velocity,
            micro.slow_velocity,
            micro.acceleration,
            micro.tick_imbalance,
            micro.trend_efficiency,
            micro.volatility_ratio,
            micro.last_move_ratio,
        )
        seconds_of_day = (tick.ts_ns // 1_000_000_000) % 86400
        phase = 2.0 * math.pi * float(seconds_of_day) / 86400.0
        spread_fraction = math.tanh((micro.spread / max(micro.mid, 1e-12)) * 100_000.0 / 5.0)
        execution = execution or ExecutionContext()

        vector: list[float] = [float(v) for v in micro_vector]
        for seconds in FRAME_SECONDS:
            vector.extend(snapshots[seconds].vector)
        vector.extend((math.sin(phase), math.cos(phase), spread_fraction))
        vector.extend(execution.vector())

        slow = snapshots[60]
        if slow.shock_ratio >= 2.75 or micro.volatility_ratio >= 3.75:
            regime = "shock"
        elif slow.atr_spreads < 1.5 and abs(micro.fast_velocity) < 0.10:
            regime = "quiet"
        elif slow.trend_strength >= 0.35 and slow.range_position >= 0.15 and slow.ema_cross_price > 0:
            regime = "trend_up"
        elif slow.trend_strength >= 0.35 and slow.range_position <= -0.15 and slow.ema_cross_price < 0:
            regime = "trend_down"
        else:
            regime = "range"

        if len(vector) != len(self.feature_names) or not np.isfinite(vector).all():
            raise ValueError("invalid encoded market state")
        context = {
            "mid": float(micro.mid),
            "spread": float(micro.spread),
            "frames": {
                str(seconds): {
                    "atr_price": snapshots[seconds].atr_price,
                    "atr_spreads": snapshots[seconds].atr_spreads,
                    "trend_strength": snapshots[seconds].trend_strength,
                    "range_position": snapshots[seconds].range_position,
                    "shock_ratio": snapshots[seconds].shock_ratio,
                }
                for seconds in FRAME_SECONDS
            },
        }
        return MarketState(
            ts_ns=tick.ts_ns,
            embedding=tuple(vector),
            regime=regime,
            feature_names=self.feature_names,
            context=context,
        )
