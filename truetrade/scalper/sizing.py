from __future__ import annotations

from dataclasses import dataclass
import math

from .risk import RiskController
from .types import MicroFeatures, PositionState, Side


@dataclass(frozen=True)
class AdaptiveSizeSettings:
    minimum_size: float = 0.01
    maximum_size: float = 0.03
    minimum_quality_multiplier: float = 0.35
    maximum_multiplier: float = 2.25

    def __post_init__(self) -> None:
        if not 0 < self.minimum_size <= self.maximum_size:
            raise ValueError("invalid size bounds")
        if not 0 < self.minimum_quality_multiplier <= 1:
            raise ValueError("invalid quality floor")
        if self.maximum_multiplier < 1:
            raise ValueError("invalid maximum multiplier")


class AdaptiveSizer:
    """Translate edge quality into requested size while preserving hard inventory caps."""

    def __init__(self, settings: AdaptiveSizeSettings | None = None):
        self.settings = settings or AdaptiveSizeSettings()

    def size(self, *, base_size: float, side: Side, confidence: float, threshold: float,
             features: MicroFeatures, positions: list[PositionState], risk: RiskController,
             quality_multiplier: float, performance_multiplier: float, adding: bool) -> float | None:
        cfg = self.settings
        if base_size <= 0 or not 0.5 < threshold < 1 or not threshold <= confidence <= 1:
            return None
        if quality_multiplier < cfg.minimum_quality_multiplier or performance_multiplier <= 0:
            return None

        confidence_score = min(1.0, max(0.0, (confidence - threshold) / max(1e-9, 1.0 - threshold)))
        edge_multiplier = 0.65 + 1.55 * confidence_score
        trend_multiplier = 0.75 + 0.75 * min(1.0, abs(features.trend_efficiency))
        volatility_multiplier = 1.0 / (1.0 + 0.35 * max(0.0, features.volatility_ratio - 1.0))
        spread_multiplier = 1.0 / (1.0 + 0.20 * max(0.0, features.spread_z))
        add_multiplier = 0.80 if adding else 1.0

        multiplier = min(
            cfg.maximum_multiplier,
            edge_multiplier * trend_multiplier * volatility_multiplier * spread_multiplier
            * quality_multiplier * performance_multiplier * add_multiplier,
        )
        raw = base_size * multiplier
        if raw < cfg.minimum_size:
            return None

        same = sum(p.size for p in positions if p.side is side)
        total = sum(p.size for p in positions)
        cap = min(cfg.maximum_size, risk.limits.max_total_size - total,
                  risk.limits.max_directional_size - same)
        if cap < cfg.minimum_size - 1e-12:
            return None
        value = min(raw, cap)
        if not math.isfinite(value) or value < cfg.minimum_size - 1e-12:
            return None
        return round(value, 8)
