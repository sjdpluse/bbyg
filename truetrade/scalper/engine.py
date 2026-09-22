from __future__ import annotations

from dataclasses import dataclass

from .features import TickFeatureEngine
from .learning import ChampionChallenger
from .position import AlgorithmicExitManager
from .risk import RiskController
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
    """Broker-neutral local decision core with event-gated entries and algorithmic exits."""

    def __init__(self, *, features=None, model=None, exits=None, risk=None, entry=None):
        self.features = features or TickFeatureEngine()
        self.model = model or ChampionChallenger()
        self.exits = exits or AlgorithmicExitManager()
        self.risk = risk or RiskController()
        self.entry = entry or EntrySettings()
        self._last_entry_ns: dict[Side, int] = {}
        self._last_entry_mid: dict[Side, float] = {}

    def _entry_event_allowed(self, side: Side, tick: Tick, spread: float, confidence: float,
                             has_same_side: bool) -> bool:
        last_ns = self._last_entry_ns.get(side)
        if last_ns is not None:
            if tick.ts_ns - last_ns < self.entry.min_entry_gap_ms * 1_000_000:
                return False
            last_mid = self._last_entry_mid[side]
            moved = abs(tick.mid - last_mid) / max(spread, 1e-12)
            if moved < self.entry.min_reentry_move_spreads and confidence < min(0.98, self.entry.add_probability + 0.08):
                return False
        if has_same_side and confidence < self.entry.add_probability:
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
        intents = [self.exits.evaluate(p, tick, f, p_long) for p in positions]
        if any(i.kind is IntentKind.CLOSE for i in intents):
            return intents
        if not self.model.qualified:
            return intents

        long_conf = p_long
        short_conf = 1.0 - p_long
        if long_conf >= self.entry.min_probability:
            side, conf = Side.LONG, long_conf
        elif short_conf >= self.entry.min_probability:
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
        if not self._entry_event_allowed(side, tick, f.spread, conf, bool(same)):
            return intents

        allowed, reason = self.risk.can_open(positions, side, self.entry.default_size, f.spread_z)
        if allowed:
            kind = IntentKind.ADD if same else IntentKind.OPEN
            intents.append(Intent(kind, "qualified_edge", side=side, fraction=1.0,
                                  confidence=conf, size=self.entry.default_size))
        else:
            intents.append(Intent(IntentKind.HOLD, reason, side=side, confidence=conf))
        return intents
