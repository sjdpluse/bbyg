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
    min_trend_efficiency: float = 0.12
    min_velocity: float = 0.05
    default_size: float = 0.01


class ScalperCore:
    """Broker-neutral local decision core.

    It emits intents only; broker writes belong in a separate execution adapter. This keeps
    the learning/strategy layer testable and prevents accidental retries or hidden writes.
    """

    def __init__(
        self,
        *,
        features: TickFeatureEngine | None = None,
        model: ChampionChallenger | None = None,
        exits: AlgorithmicExitManager | None = None,
        risk: RiskController | None = None,
        entry: EntrySettings | None = None,
    ):
        self.features = features or TickFeatureEngine()
        self.model = model or ChampionChallenger()
        self.exits = exits or AlgorithmicExitManager()
        self.risk = risk or RiskController()
        self.entry = entry or EntrySettings()

    def on_tick(self, tick: Tick, positions: list[PositionState]) -> list[Intent]:
        f = self.features.update(tick)
        if f is None:
            return []
        p_long = self.model.probability_long(f)
        intents = [self.exits.evaluate(p, tick, f, p_long) for p in positions]
        if any(i.kind is IntentKind.CLOSE for i in intents):
            return intents

        # No autonomous entry before a candidate has passed the independent promotion gate.
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

        allowed, reason = self.risk.can_open(positions, side, self.entry.default_size, f.spread_z)
        if allowed:
            intents.append(Intent(IntentKind.OPEN, "qualified_edge", side=side,
                                  fraction=1.0, confidence=conf))
        else:
            intents.append(Intent(IntentKind.HOLD, reason, side=side, confidence=conf))
        return intents
