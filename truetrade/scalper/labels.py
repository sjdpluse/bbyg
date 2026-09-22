from __future__ import annotations

from dataclasses import dataclass

from .types import Tick


@dataclass(frozen=True)
class LabelSettings:
    profit_spreads: float = 1.6
    loss_spreads: float = 1.4
    extra_cost_spreads: float = 0.20
    max_lookahead_ticks: int = 600

    def __post_init__(self) -> None:
        if self.profit_spreads <= 0 or self.loss_spreads <= 0 or self.extra_cost_spreads < 0:
            raise ValueError("invalid barrier distances")
        if self.max_lookahead_ticks < 10:
            raise ValueError("max_lookahead_ticks too small")


@dataclass(frozen=True)
class LabelOutcome:
    label: int | None
    reason: str
    ticks_observed: int


class CostAwareLabeler:
    """Event-based directional labeler using executable bid/ask prices.

    max_lookahead_ticks only decides whether a historical sample has enough future
    evidence. It never creates a time-based trading exit. If neither directional target
    wins inside the evidence window, the sample remains unlabeled.
    """

    def __init__(self, settings: LabelSettings | None = None):
        self.settings = settings or LabelSettings()

    def outcome(self, anchor: Tick, future: list[Tick]) -> LabelOutcome:
        s = self.settings
        if not future:
            return LabelOutcome(None, "no_future_ticks", 0)
        spread = max(anchor.spread, 1e-12)
        long_entry = anchor.ask
        short_entry = anchor.bid
        long_profit = long_entry + (s.profit_spreads + s.extra_cost_spreads) * spread
        long_loss = anchor.bid - s.loss_spreads * spread
        short_profit = short_entry - (s.profit_spreads + s.extra_cost_spreads) * spread
        short_loss = anchor.ask + s.loss_spreads * spread

        for i, tick in enumerate(future[: s.max_lookahead_ticks], start=1):
            long_win = tick.bid >= long_profit
            short_win = tick.ask <= short_profit
            long_fail = tick.bid <= long_loss
            short_fail = tick.ask >= short_loss
            if long_win and short_win:
                return LabelOutcome(None, "ambiguous_simultaneous_profit", i)
            if long_win and not short_win:
                return LabelOutcome(1, "long_net_target_first", i)
            if short_win and not long_win:
                return LabelOutcome(0, "short_net_target_first", i)
            if long_fail and short_fail:
                return LabelOutcome(None, "both_directions_adverse", i)
        return LabelOutcome(None, "unresolved_path", min(len(future), s.max_lookahead_ticks))

    def label(self, anchor: Tick, future: list[Tick]) -> int | None:
        return self.outcome(anchor, future).label
