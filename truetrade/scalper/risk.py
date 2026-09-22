from __future__ import annotations

from dataclasses import dataclass
import time

from .types import PositionState, Side


@dataclass(frozen=True)
class ScalperRiskLimits:
    max_positions: int = 6
    max_same_side_positions: int = 4
    max_total_size: float = 0.10
    max_directional_size: float = 0.08
    max_orders_per_second: int = 4
    max_spread_z: float = 4.0

    def __post_init__(self) -> None:
        if self.max_positions < 1 or self.max_same_side_positions < 1:
            raise ValueError("position limits must be positive")
        if self.max_same_side_positions > self.max_positions:
            raise ValueError("same-side limit cannot exceed total limit")
        if self.max_total_size <= 0 or self.max_directional_size <= 0:
            raise ValueError("size limits must be positive")
        if self.max_orders_per_second < 1:
            raise ValueError("order rate limit must be positive")


class RiskController:
    def __init__(self, limits: ScalperRiskLimits | None = None):
        self.limits = limits or ScalperRiskLimits()
        self._order_times: list[float] = []

    def _trim(self, now: float) -> None:
        self._order_times = [x for x in self._order_times if now - x < 1.0]

    def note_order(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._trim(now)
        self._order_times.append(now)

    def can_open(self, positions: list[PositionState], side: Side, size: float, spread_z: float,
                 now: float | None = None) -> tuple[bool, str]:
        now = time.monotonic() if now is None else now
        self._trim(now)
        if spread_z > self.limits.max_spread_z:
            return False, "spread_regime_blocked"
        if len(self._order_times) >= self.limits.max_orders_per_second:
            return False, "order_rate_limited"
        if len(positions) >= self.limits.max_positions:
            return False, "position_limit"
        same = [p for p in positions if p.side is side]
        if len(same) >= self.limits.max_same_side_positions:
            return False, "same_side_position_limit"
        total = sum(p.size for p in positions) + size
        if total > self.limits.max_total_size + 1e-12:
            return False, "total_exposure_limit"
        directional = sum(p.size for p in same) + size
        if directional > self.limits.max_directional_size + 1e-12:
            return False, "directional_exposure_limit"
        return True, "ok"
