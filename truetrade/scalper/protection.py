from __future__ import annotations

from dataclasses import dataclass

from .types import Intent, IntentKind, MicroFeatures, PositionState, Side, Tick


@dataclass(frozen=True)
class ProtectionSettings:
    activation_spreads: float = 1.5
    breakeven_lock_spreads: float = 0.20
    trail_distance_spreads: float = 1.10
    strong_edge_trail_spreads: float = 1.80
    strong_edge_threshold: float = 0.32
    min_tighten_spreads: float = 0.20
    market_buffer_spreads: float = 0.35

    def __post_init__(self) -> None:
        values = (
            self.activation_spreads, self.trail_distance_spreads,
            self.strong_edge_trail_spreads, self.min_tighten_spreads,
            self.market_buffer_spreads,
        )
        if any(x <= 0 for x in values):
            raise ValueError("protection distances must be positive")


class DynamicProtectionManager:
    """Tighten broker-side emergency protection only when market state earns it."""

    def __init__(self, settings: ProtectionSettings | None = None):
        self.settings = settings or ProtectionSettings()

    def evaluate(self, position: PositionState, tick: Tick, features: MicroFeatures,
                 probability_long: float) -> Intent | None:
        s = self.settings
        spread = max(features.spread, 1e-12)
        directional_edge = ((probability_long - 0.5) * 2.0) * position.side.sign

        if position.side is Side.LONG:
            peak = position.peak_exit_price if position.peak_exit_price is not None else tick.bid
            favorable = peak - position.entry
            if favorable / spread < s.activation_spreads:
                return None
            lock = position.entry + s.breakeven_lock_spreads * spread
            trail = s.strong_edge_trail_spreads if directional_edge >= s.strong_edge_threshold else s.trail_distance_spreads
            candidate = max(lock, peak - trail * spread)
            candidate = min(candidate, tick.bid - s.market_buffer_spreads * spread)
            current = position.broker_stop
            if current is not None and candidate <= current + s.min_tighten_spreads * spread:
                return None
        else:
            peak = position.peak_exit_price if position.peak_exit_price is not None else tick.ask
            favorable = position.entry - peak
            if favorable / spread < s.activation_spreads:
                return None
            lock = position.entry - s.breakeven_lock_spreads * spread
            trail = s.strong_edge_trail_spreads if directional_edge >= s.strong_edge_threshold else s.trail_distance_spreads
            candidate = min(lock, peak + trail * spread)
            candidate = max(candidate, tick.ask + s.market_buffer_spreads * spread)
            current = position.broker_stop
            if current is not None and candidate >= current - s.min_tighten_spreads * spread:
                return None

        if candidate <= 0:
            return None
        return Intent(IntentKind.PROTECT, "dynamic_edge_trailing_stop",
                      position_id=position.position_id,
                      confidence=min(1.0, max(0.0, directional_edge)), stop=float(candidate))
