from __future__ import annotations

from collections import deque
import math

import numpy as np

from .types import MicroFeatures, Tick


class TickFeatureEngine:
    """Causal tick-level microstructure features normalized by current transaction cost.

    The engine never looks forward. Most directional features are measured in spread-units,
    which keeps the scale meaningful as XAUUSD price and spread regimes change.
    """

    def __init__(self, window: int = 96, min_ticks: int = 24, fast_ticks: int = 8, slow_ticks: int = 24):
        if not 24 <= window <= 4096:
            raise ValueError("window out of range")
        if not 8 <= min_ticks <= window:
            raise ValueError("min_ticks out of range")
        if not 2 <= fast_ticks < slow_ticks <= window:
            raise ValueError("invalid velocity windows")
        self.ticks: deque[Tick] = deque(maxlen=window)
        self.min_ticks = min_ticks
        self.fast_ticks = fast_ticks
        self.slow_ticks = slow_ticks

    def update(self, tick: Tick) -> MicroFeatures | None:
        if self.ticks and tick.ts_ns <= self.ticks[-1].ts_ns:
            raise ValueError("ticks must be strictly increasing")
        self.ticks.append(tick)
        if len(self.ticks) < self.min_ticks:
            return None

        ticks = list(self.ticks)
        mids = np.asarray([t.mid for t in ticks], dtype=float)
        spreads = np.asarray([max(t.spread, 1e-12) for t in ticks], dtype=float)
        times = np.asarray([t.ts_ns for t in ticks], dtype=np.float64) / 1e9
        current_spread = float(spreads[-1])

        diffs = np.diff(mids)
        dt = np.maximum(np.diff(times), 1e-6)
        speed = diffs / dt

        def velocity(last_n: int) -> float:
            n = min(last_n, len(speed))
            if n <= 0:
                return 0.0
            return float(np.mean(speed[-n:]) / current_spread)

        fast = velocity(self.fast_ticks)
        slow = velocity(self.slow_ticks)
        acceleration = fast - slow

        signs = np.sign(diffs[-min(self.slow_ticks, len(diffs)):])
        imbalance = float(np.mean(signs)) if len(signs) else 0.0

        recent = diffs[-min(self.slow_ticks, len(diffs)):]
        gross = float(np.abs(recent).sum())
        net = float(recent.sum())
        efficiency = 0.0 if gross <= 1e-12 else float(net / gross)

        noise = float(np.sqrt(np.mean(recent * recent))) if len(recent) else 0.0
        volatility_ratio = noise / current_spread
        last_move_ratio = float(diffs[-1] / current_spread) if len(diffs) else 0.0

        hist = spreads[:-1] if len(spreads) > 1 else spreads
        median = float(np.median(hist))
        mad = float(np.median(np.abs(hist - median)))
        robust_scale = max(1.4826 * mad, median * 1e-6, 1e-12)
        spread_z = float((current_spread - median) / robust_scale)

        values = (fast, slow, acceleration, imbalance, efficiency, volatility_ratio, last_move_ratio, spread_z)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("non-finite feature")

        return MicroFeatures(
            ts_ns=tick.ts_ns,
            mid=tick.mid,
            spread=current_spread,
            spread_z=spread_z,
            fast_velocity=fast,
            slow_velocity=slow,
            acceleration=acceleration,
            tick_imbalance=imbalance,
            trend_efficiency=efficiency,
            volatility_ratio=volatility_ratio,
            last_move_ratio=last_move_ratio,
            noise_price=noise,
        )
