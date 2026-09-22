from __future__ import annotations

from collections import deque
import math

import numpy as np

from .types import MicroFeatures, Tick


class TickFeatureEngine:
    """Causal tick-level microstructure features with robust bounded scaling.

    Historical MT5 streams can contain several quote changes with the same millisecond
    timestamp. Dividing those moves by an artificial sub-millisecond ``dt`` creates
    enormous pseudo-velocities that do not represent tradable information. BBYG therefore
    measures directional velocity in *spread-normalized move per tick* rather than raw
    price-per-second. This keeps historical replay and live inference on the same stable
    event-time representation.
    """

    MOVE_CLIP = 8.0
    SPREAD_Z_CLIP = 8.0
    VOLATILITY_CLIP = 4.0

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
        current_spread = float(spreads[-1])

        diffs = np.diff(mids)
        # Normalize each quote move by the spread observable at that move.  Clipping at
        # the event level prevents bad/duplicate timestamps or isolated quote jumps from
        # dominating an entire learning window.
        local_spreads = np.maximum(spreads[1:], 1e-12)
        move_units = np.clip(diffs / local_spreads, -self.MOVE_CLIP, self.MOVE_CLIP)

        def velocity(last_n: int) -> float:
            n = min(last_n, len(move_units))
            if n <= 0:
                return 0.0
            return float(np.mean(move_units[-n:]))

        fast = velocity(self.fast_ticks)
        slow = velocity(self.slow_ticks)
        acceleration = float(np.clip(fast - slow, -self.MOVE_CLIP, self.MOVE_CLIP))

        signs = np.sign(diffs[-min(self.slow_ticks, len(diffs)):])
        imbalance = float(np.mean(signs)) if len(signs) else 0.0

        recent = diffs[-min(self.slow_ticks, len(diffs)):]
        gross = float(np.abs(recent).sum())
        net = float(recent.sum())
        efficiency = 0.0 if gross <= 1e-12 else float(np.clip(net / gross, -1.0, 1.0))

        noise = float(np.sqrt(np.mean(recent * recent))) if len(recent) else 0.0
        volatility_ratio = float(np.clip(noise / current_spread, 0.0, self.VOLATILITY_CLIP))
        last_move_ratio = (
            float(np.clip(diffs[-1] / current_spread, -self.MOVE_CLIP, self.MOVE_CLIP))
            if len(diffs) else 0.0
        )

        hist = spreads[:-1] if len(spreads) > 1 else spreads
        median = float(np.median(hist))
        mad = float(np.median(np.abs(hist - median)))
        # A nearly fixed-spread feed makes MAD ~0.  Falling back to 2% of the median
        # avoids turning tiny floating-point differences into huge z-scores.
        robust_scale = max(1.4826 * mad, median * 0.02, 1e-12)
        spread_z = float(np.clip((current_spread - median) / robust_scale,
                                 -self.SPREAD_Z_CLIP, self.SPREAD_Z_CLIP))

        values = (fast, slow, acceleration, imbalance, efficiency,
                  volatility_ratio, last_move_ratio, spread_z)
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
