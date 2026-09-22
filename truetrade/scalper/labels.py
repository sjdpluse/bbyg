from __future__ import annotations

from dataclasses import dataclass

from .types import Tick


@dataclass(frozen=True)
class LabelSettings:
    profit_spreads: float = 1.6
    loss_spreads: float = 1.4
    extra_cost_spreads: float = 0.20
    max_lookahead_ticks: int = 600
    max_entry_delay_seconds: float = 5.0
    stop_reference: str = "exit_quote"

    def __post_init__(self) -> None:
        if self.profit_spreads <= 0 or self.loss_spreads <= 0 or self.extra_cost_spreads < 0:
            raise ValueError("invalid barrier distances")
        if self.max_lookahead_ticks < 10:
            raise ValueError("max_lookahead_ticks too small")
        if self.max_entry_delay_seconds <= 0:
            raise ValueError("max_entry_delay_seconds must be positive")
        if self.stop_reference not in {"exit_quote", "entry"}:
            raise ValueError("stop_reference must be 'exit_quote' or 'entry'")

    def long_stop_price(self, bid, ask, spread):
        base = ask if self.stop_reference == "entry" else bid
        return base - self.loss_spreads * spread

    def short_stop_price(self, bid, ask, spread):
        base = bid if self.stop_reference == "entry" else ask
        return base + self.loss_spreads * spread

    @property
    def nominal_target_from_entry_spreads(self) -> float:
        return self.profit_spreads + self.extra_cost_spreads

    @property
    def nominal_stop_from_entry_spreads(self) -> float:
        return self.loss_spreads if self.stop_reference == "entry" else self.loss_spreads + 1.0


@dataclass(frozen=True)
class LabelOutcome:
    label: int | None
    reason: str
    ticks_observed: int


class CostAwareLabeler:
    """Executable first-passage directional labeler using bid/ask prices.

    The ``anchor`` passed to :meth:`outcome` is the executable entry quote, not the
    earlier feature/decision tick. Replay builders are responsible for mapping a causal
    decision at tick *t* to the first strictly later executable tick before calling this
    labeler. This keeps offline labels aligned with demo/live order timing.

    Each direction is allowed to produce a label only while that hypothetical trade is
    still alive. If its executable stop is crossed before its target, later recovery to
    that target cannot relabel the path as a winner.

    ``stop_reference='exit_quote'`` preserves the original BBYG research contract, where
    the stop is measured beyond the current executable exit quote. ``'entry'`` is an
    explicit research alternative where ``loss_spreads`` is the actual entry-to-stop
    distance.

    ``max_lookahead_ticks`` counts executable ticks after entry. It limits evidence only;
    it does not itself create a PnL result for unresolved paths.
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
        long_stop = s.long_stop_price(anchor.bid, anchor.ask, spread)
        short_profit = short_entry - (s.profit_spreads + s.extra_cost_spreads) * spread
        short_stop = s.short_stop_price(anchor.bid, anchor.ask, spread)

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
