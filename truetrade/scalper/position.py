from __future__ import annotations

from dataclasses import dataclass

from .types import Intent, IntentKind, MicroFeatures, PositionState, Side, Tick


@dataclass(frozen=True)
class ExitSettings:
    hard_stop_spreads: float = 3.0
    profit_lock_activation_spreads: float = 2.0
    trail_backoff_spreads: float = 1.0
    close_edge_reversal: float = -0.18
    reduce_edge_below: float = 0.08
    keep_edge_above: float = 0.14
    reduce_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.hard_stop_spreads <= 0 or self.profit_lock_activation_spreads <= 0 or self.trail_backoff_spreads <= 0:
            raise ValueError("distance settings must be positive")
        if not 0 < self.reduce_fraction < 1:
            raise ValueError("reduce_fraction must be between zero and one")


class AlgorithmicExitManager:
    """Pure market-state exit logic. There is intentionally no elapsed-time exit."""

    def __init__(self, settings: ExitSettings | None = None):
        self.settings = settings or ExitSettings()

    @staticmethod
    def _exit_price(position: PositionState, tick: Tick) -> float:
        return tick.bid if position.side is Side.LONG else tick.ask

    def evaluate(
        self,
        position: PositionState,
        tick: Tick,
        features: MicroFeatures,
        probability_long: float,
    ) -> Intent:
        s = self.settings
        side_sign = position.side.sign
        exit_price = self._exit_price(position, tick)
        directional_pnl = (exit_price - position.entry) * side_sign
        spread = max(features.spread, 1e-12)
        pnl_spreads = directional_pnl / spread

        if position.side is Side.LONG:
            position.peak_exit_price = max(position.peak_exit_price or exit_price, exit_price)
            position.trough_exit_price = min(position.trough_exit_price or exit_price, exit_price)
            favorable = position.peak_exit_price - position.entry
            giveback = position.peak_exit_price - exit_price
        else:
            position.peak_exit_price = min(position.peak_exit_price or exit_price, exit_price)
            position.trough_exit_price = max(position.trough_exit_price or exit_price, exit_price)
            favorable = position.entry - position.peak_exit_price
            giveback = exit_price - position.peak_exit_price

        directional_edge = ((probability_long - 0.5) * 2.0) * side_sign
        momentum = features.fast_velocity * side_sign
        trend = features.trend_efficiency * side_sign
        imbalance = features.tick_imbalance * side_sign

        if pnl_spreads <= -s.hard_stop_spreads:
            return Intent(IntentKind.CLOSE, "hard_risk_stop", position_id=position.position_id, confidence=1.0)

        if directional_edge <= s.close_edge_reversal and (momentum < 0 or imbalance < -0.15):
            return Intent(IntentKind.CLOSE, "edge_reversal", position_id=position.position_id,
                          confidence=min(1.0, abs(directional_edge)))

        activated = favorable / spread >= s.profit_lock_activation_spreads
        trailed = giveback / spread >= s.trail_backoff_spreads
        if activated and trailed and directional_edge < s.keep_edge_above:
            return Intent(IntentKind.CLOSE, "algorithmic_profit_lock", position_id=position.position_id,
                          confidence=min(1.0, max(0.0, pnl_spreads / 4.0)))

        weakening = directional_edge < s.reduce_edge_below and (momentum <= 0 or trend <= 0)
        if pnl_spreads > 0 and weakening and position.reductions == 0:
            position.reductions += 1
            return Intent(IntentKind.REDUCE, "edge_decay_partial", position_id=position.position_id,
                          fraction=s.reduce_fraction, confidence=min(1.0, max(0.0, pnl_spreads / 3.0)))

        return Intent(IntentKind.HOLD, "edge_intact", position_id=position.position_id,
                      confidence=min(1.0, max(0.0, directional_edge)))
