from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EconomicThresholdCandidate:
    threshold: float
    trades: int
    net_pnl_spreads: float
    average_net_pnl_spreads: float | None
    profit_factor: float | None
    win_rate: float | None

    @property
    def economically_positive(self) -> bool:
        return (
            self.trades > 0
            and self.net_pnl_spreads > 0.0
            and self.average_net_pnl_spreads is not None
            and self.average_net_pnl_spreads > 0.0
            and self.profit_factor is not None
            and self.profit_factor > 1.0
        )


def choose_economic_threshold(
    candidates: list[EconomicThresholdCandidate],
    *,
    min_trades: int = 100,
    min_profit_factor: float = 1.05,
) -> EconomicThresholdCandidate | None:
    """Choose a calibration-only threshold or deliberately abstain.

    The gate is fail-closed: if calibration cannot show positive trade-level expectancy
    with enough executions, no threshold is returned. Among eligible candidates we favor
    total net expectancy first, then profit factor, then average expectancy, and finally
    the higher confidence threshold. Validation/holdout data must never be passed here.
    """
    if min_trades < 1 or min_profit_factor <= 1.0:
        raise ValueError("invalid economic gate settings")
    eligible = [
        c for c in candidates
        if c.trades >= min_trades
        and c.economically_positive
        and c.profit_factor is not None
        and c.profit_factor >= min_profit_factor
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda c: (
            c.net_pnl_spreads,
            float(c.profit_factor),
            float(c.average_net_pnl_spreads),
            c.threshold,
        ),
    )
