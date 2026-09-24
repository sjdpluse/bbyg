from __future__ import annotations

from dataclasses import dataclass
import math

from .experience import RewardComponents, bounded_reward
from .types import Side, Tick


@dataclass(frozen=True)
class CounterfactualSettings:
    horizon_ticks: int = 600
    risk_spreads: float = 2.0
    extra_cost_spreads: float = 0.20

    def __post_init__(self) -> None:
        if self.horizon_ticks < 10:
            raise ValueError("horizon_ticks too small")
        if self.risk_spreads <= 0 or self.extra_cost_spreads < 0:
            raise ValueError("invalid counterfactual settings")


@dataclass(frozen=True)
class SideOutcome:
    side: Side
    entry_price: float
    exit_price: float
    net_r: float
    mfe_r: float
    mae_r: float
    execution_cost_r: float
    reward: float

    def __post_init__(self) -> None:
        values = (
            self.entry_price, self.exit_price, self.net_r, self.mfe_r,
            self.mae_r, self.execution_cost_r, self.reward,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("counterfactual outcome must be finite")


@dataclass(frozen=True)
class CounterfactualOutcome:
    anchor_ts_ns: int
    resolved_ts_ns: int
    ticks_observed: int
    long: SideOutcome
    short: SideOutcome


@dataclass
class _Pending:
    anchor: Tick
    future: list[Tick]


class CounterfactualEpisodeResolver:
    """Resolve both LONG and SHORT outcomes for every eligible market state.

    The resolver is policy-independent: executed, rejected and FLAT decisions all receive
    the same future-path treatment.  Entry occurs at the first strictly later executable
    quote.  No future information is used until `advance()` receives that tick.
    """

    def __init__(self, settings: CounterfactualSettings | None = None):
        self.settings = settings or CounterfactualSettings()
        self.pending: dict[int, _Pending] = {}

    def add(self, anchor: Tick) -> None:
        if anchor.ts_ns in self.pending:
            raise ValueError("duplicate counterfactual anchor")
        self.pending[anchor.ts_ns] = _Pending(anchor, [])

    @staticmethod
    def _reward(side: Side, entry: Tick, future: list[Tick], settings: CounterfactualSettings) -> SideOutcome:
        spread = max(entry.spread, 1e-12)
        risk_price = settings.risk_spreads * spread
        if side is Side.LONG:
            entry_price = entry.ask
            exits = [t.bid for t in future]
            gross_moves = [price - entry_price for price in exits]
        else:
            entry_price = entry.bid
            exits = [t.ask for t in future]
            gross_moves = [entry_price - price for price in exits]
        if not gross_moves:
            raise ValueError("future path required")

        terminal_move = gross_moves[-1]
        mfe = max(0.0, max(gross_moves))
        mae = max(0.0, -min(gross_moves))
        cost_r = settings.extra_cost_spreads / settings.risk_spreads
        net_r = terminal_move / risk_price - cost_r
        mfe_r = mfe / risk_price
        mae_r = mae / risk_price
        regret_r = max(0.0, mfe_r - max(net_r, 0.0))
        components = RewardComponents(
            net_r=float(net_r),
            mfe_r=float(mfe_r),
            mae_r=float(mae_r),
            execution_cost_r=float(cost_r),
            regret_r=float(regret_r),
        )
        return SideOutcome(
            side=side,
            entry_price=float(entry_price),
            exit_price=float(exits[-1]),
            net_r=float(net_r),
            mfe_r=float(mfe_r),
            mae_r=float(mae_r),
            execution_cost_r=float(cost_r),
            reward=bounded_reward(components),
        )

    def advance(self, tick: Tick) -> list[CounterfactualOutcome]:
        resolved: list[CounterfactualOutcome] = []
        remove: list[int] = []
        for anchor_ts, item in list(self.pending.items()):
            if tick.ts_ns <= item.anchor.ts_ns:
                continue
            item.future.append(tick)
            if len(item.future) < self.settings.horizon_ticks:
                continue
            entry = item.future[0]
            path = item.future[1:]
            if not path:
                continue
            long = self._reward(Side.LONG, entry, path, self.settings)
            short = self._reward(Side.SHORT, entry, path, self.settings)
            resolved.append(CounterfactualOutcome(
                anchor_ts_ns=anchor_ts,
                resolved_ts_ns=tick.ts_ns,
                ticks_observed=len(item.future),
                long=long,
                short=short,
            ))
            remove.append(anchor_ts)
        for anchor_ts in remove:
            self.pending.pop(anchor_ts, None)
        return resolved

    def pending_count(self) -> int:
        return len(self.pending)
