from __future__ import annotations

from dataclasses import dataclass

from .risk import ScalperRiskLimits
from .types import Intent, IntentKind, PositionState, Side, Tick


@dataclass(frozen=True)
class PortfolioSnapshot:
    positions: int
    gross_size: float
    long_size: float
    short_size: float
    concentration: float
    marked_price_pnl: float


class PortfolioExposureEngine:
    """Portfolio-level guardrail independent from the entry model."""

    def __init__(self, limits: ScalperRiskLimits):
        self.limits = limits

    def snapshot(self, positions: list[PositionState], tick: Tick) -> PortfolioSnapshot:
        long_size = sum(p.size for p in positions if p.side is Side.LONG)
        short_size = sum(p.size for p in positions if p.side is Side.SHORT)
        gross = long_size + short_size
        concentration = 0.0 if gross <= 0 else max(long_size, short_size) / gross
        pnl = 0.0
        for p in positions:
            mark = tick.bid if p.side is Side.LONG else tick.ask
            pnl += (mark - p.entry) * p.side.sign * p.size
        return PortfolioSnapshot(len(positions), gross, long_size, short_size, concentration, pnl)

    def emergency_deleveraging(self, positions: list[PositionState], tick: Tick,
                               probability_long: float) -> list[Intent]:
        if not positions:
            return []
        snap = self.snapshot(positions, tick)
        over = (
            snap.positions > self.limits.max_positions
            or snap.gross_size > self.limits.max_total_size + 1e-12
            or snap.long_size > self.limits.max_directional_size + 1e-12
            or snap.short_size > self.limits.max_directional_size + 1e-12
        )
        if not over:
            return []

        edge = (probability_long - 0.5) * 2.0
        candidates = sorted(positions, key=lambda p: edge * p.side.sign)
        intents: list[Intent] = []
        working = list(positions)
        for p in candidates:
            snap = self.snapshot(working, tick)
            bad = (
                len(working) > self.limits.max_positions
                or snap.gross_size > self.limits.max_total_size + 1e-12
                or snap.long_size > self.limits.max_directional_size + 1e-12
                or snap.short_size > self.limits.max_directional_size + 1e-12
            )
            if not bad:
                break
            intents.append(Intent(IntentKind.REDUCE, "portfolio_exposure_delever",
                                  position_id=p.position_id, fraction=0.5, confidence=1.0))
            replacement = PositionState(
                p.position_id, p.side, p.size * 0.5, p.entry, p.opened_ns,
                p.peak_exit_price, p.trough_exit_price, p.reductions, p.broker_stop,
            )
            working = [replacement if x.position_id == p.position_id else x for x in working]
        return intents
