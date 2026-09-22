from __future__ import annotations

from dataclasses import dataclass

from .features import TickFeatureEngine
from .learning import ChampionChallenger
from .portfolio import PortfolioExposureEngine
from .position import AlgorithmicExitManager
from .protection import DynamicProtectionManager
from .risk import RiskController
from .sizing import AdaptiveSizer
from .types import Intent, IntentKind, PositionState, Side, Tick


@dataclass(frozen=True)
class EntrySettings:
    min_probability: float = 0.62
    add_probability: float = 0.70
    min_trend_efficiency: float = 0.12
    min_velocity: float = 0.05
    default_size: float = 0.01
    min_entry_gap_ms: int = 120
    min_reentry_move_spreads: float = 0.30

    def __post_init__(self) -> None:
        if not 0.5 < self.min_probability < 1 or not self.min_probability <= self.add_probability < 1:
            raise ValueError("invalid probability thresholds")
        if self.default_size <= 0 or self.min_entry_gap_ms < 0 or self.min_reentry_move_spreads < 0:
            raise ValueError("invalid entry settings")


class ScalperCore:
    """Broker-neutral local decision core with adaptive sizing and market-state exits."""

    def __init__(self, *, features=None, model=None, exits=None, risk=None, entry=None,
                 sizer=None, protection=None, portfolio=None):
        self.features = features or TickFeatureEngine()
        self.model = model or ChampionChallenger()
        self.exits = exits or AlgorithmicExitManager()
        self.risk = risk or RiskController()
        self.entry = entry or EntrySettings()
        self.sizer = sizer or AdaptiveSizer()
        self.protection = protection or DynamicProtectionManager()
        self.portfolio = portfolio or PortfolioExposureEngine(self.risk.limits)
        self._last_entry_ns: dict[Side, int] = {}
        self._last_entry_mid: dict[Side, float] = {}
        self.execution_size_multiplier = 1.0
        self.execution_probability_penalty = 0.0
        self.performance_multiplier = 0.50
        self.execution_blocked = False

    def set_execution_context(self, *, size_multiplier: float, probability_penalty: float,
                              performance_multiplier: float, blocked: bool) -> None:
        self.execution_size_multiplier = min(1.0, max(0.0, float(size_multiplier)))
        self.execution_probability_penalty = min(0.20, max(0.0, float(probability_penalty)))
        self.performance_multiplier = min(1.0, max(0.0, float(performance_multiplier)))
        self.execution_blocked = bool(blocked)

    def _entry_event_allowed(self, side: Side, tick: Tick, spread: float, confidence: float,
                             has_same_side: bool, add_threshold: float) -> bool:
        last_ns = self._last_entry_ns.get(side)
        if last_ns is not None:
            if tick.ts_ns - last_ns < self.entry.min_entry_gap_ms * 1_000_000:
                return False
            last_mid = self._last_entry_mid[side]
            moved = abs(tick.mid - last_mid) / max(spread, 1e-12)
            if moved < self.entry.min_reentry_move_spreads and confidence < min(0.98, add_threshold + 0.08):
                return False
        if has_same_side and confidence < add_threshold:
            return False
        return True

    def note_entry(self, side: Side, tick: Tick) -> None:
        self._last_entry_ns[side] = tick.ts_ns
        self._last_entry_mid[side] = tick.mid

    def on_tick(self, tick: Tick, positions: list[PositionState]) -> list[Intent]:
        f = self.features.update(tick)
        if f is None:
            return []
        p_long = self.model.probability_long(f)

        portfolio_intents = self.portfolio.emergency_deleveraging(positions, tick, p_long)
        if portfolio_intents:
            return portfolio_intents

        intents: list[Intent] = []
        for position in positions:
            exit_intent = self.exits.evaluate(position, tick, f, p_long)
            if exit_intent.kind is IntentKind.HOLD:
                protection = self.protection.evaluate(position, tick, f, p_long)
                intents.append(protection or exit_intent)
            else:
                intents.append(exit_intent)

        if any(i.kind is IntentKind.CLOSE for i in intents):
            return intents
        if not self.model.qualified or self.execution_blocked:
            return intents

        threshold = min(0.92, self.entry.min_probability + self.execution_probability_penalty)
        add_threshold = min(0.96, self.entry.add_probability + self.execution_probability_penalty)
        long_conf = p_long
        short_conf = 1.0 - p_long
        if long_conf >= threshold:
            side, conf = Side.LONG, long_conf
        elif short_conf >= threshold:
            side, conf = Side.SHORT, short_conf
        else:
            return intents

        sign = side.sign
        if f.trend_efficiency * sign < self.entry.min_trend_efficiency:
            return intents
        if f.fast_velocity * sign < self.entry.min_velocity:
            return intents
        if any(p.side is not side for p in positions):
            return intents
        same = [p for p in positions if p.side is side]
        if not self._entry_event_allowed(side, tick, f.spread, conf, bool(same), add_threshold):
            return intents

        size = self.sizer.size(
            base_size=self.entry.default_size, side=side, confidence=conf,
            threshold=add_threshold if same else threshold, features=f, positions=positions,
            risk=self.risk, quality_multiplier=self.execution_size_multiplier,
            performance_multiplier=self.performance_multiplier, adding=bool(same),
        )
        if size is None:
            return intents

        allowed, reason = self.risk.can_open(positions, side, size, f.spread_z)
        if allowed:
            kind = IntentKind.ADD if same else IntentKind.OPEN
            intents.append(Intent(kind, "qualified_adaptive_edge", side=side, fraction=1.0,
                                  confidence=conf, size=size))
        else:
            intents.append(Intent(IntentKind.HOLD, reason, side=side, confidence=conf))
        return intents
