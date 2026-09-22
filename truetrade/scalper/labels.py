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
    """Executable first-passage directional labeler using bid/ask prices.

    Each direction is allowed to produce a label only while that hypothetical trade is
    still alive. If its executable stop is crossed before its target, later recovery to
    that target cannot relabel the path as a winner. This is essential for short-horizon
    trading research: a stopped-out trade is not converted into a historical winner by a
    later price reversal.

    ``max_lookahead_ticks`` limits evidence only; it does not create a time-based exit.
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
        long_stop = anchor.bid - s.loss_spreads * spread
        short_profit = short_entry - (s.profit_spreads + s.extra_cost_spreads) * spread
        short_stop = anchor.ask + s.loss_spreads * spread

        long_alive = True
        short_alive = True
        for i, tick in enumerate(future[: s.max_lookahead_ticks], start=1):
            long_win = long_alive and tick.bid >= long_profit
            short_win = short_alive and tick.ask <= short_profit
            if long_win and short_win:
                return LabelOutcome(None, "ambiguous_simultaneous_profit", i)
            if long_win:
                return LabelOutcome(1, "long_target_before_stop", i)
            if short_win:
                return LabelOutcome(0, "short_target_before_stop", i)

            if long_alive and tick.bid <= long_stop:
                long_alive = False
            if short_alive and tick.ask >= short_stop:
                short_alive = False
            if not long_alive and not short_alive:
                return LabelOutcome(None, "both_directions_stopped", i)

        return LabelOutcome(None, "unresolved_path", min(len(future), s.max_lookahead_ticks))

    def label(self, anchor: Tick, future: list[Tick]) -> int | None:
        return self.outcome(anchor, future).label
