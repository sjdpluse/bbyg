from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        return 1 if self is Side.LONG else -1


class IntentKind(str, Enum):
    OPEN = "OPEN"
    ADD = "ADD"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Tick:
    ts_ns: int
    bid: float
    ask: float
    last: float = 0.0
    volume: float = 0.0

    def __post_init__(self) -> None:
        if self.ts_ns <= 0:
            raise ValueError("tick timestamp must be positive")
        if not all(math.isfinite(x) for x in (self.bid, self.ask, self.last, self.volume)):
            raise ValueError("tick fields must be finite")
        if self.bid <= 0 or self.ask <= 0 or self.ask < self.bid:
            raise ValueError("invalid bid/ask")
        if self.volume < 0:
            raise ValueError("volume cannot be negative")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True)
class MicroFeatures:
    ts_ns: int
    mid: float
    spread: float
    spread_z: float
    fast_velocity: float
    slow_velocity: float
    acceleration: float
    tick_imbalance: float
    trend_efficiency: float
    volatility_ratio: float
    last_move_ratio: float
    noise_price: float

    def vector(self) -> tuple[float, ...]:
        return (
            self.spread_z,
            self.fast_velocity,
            self.slow_velocity,
            self.acceleration,
            self.tick_imbalance,
            self.trend_efficiency,
            self.volatility_ratio,
            self.last_move_ratio,
        )


@dataclass
class PositionState:
    position_id: str
    side: Side
    size: float
    entry: float
    opened_ns: int
    peak_exit_price: float | None = None
    trough_exit_price: float | None = None
    reductions: int = 0

    def __post_init__(self) -> None:
        if not self.position_id:
            raise ValueError("position_id required")
        if self.size <= 0 or not math.isfinite(self.size):
            raise ValueError("position size must be positive and finite")
        if self.entry <= 0 or not math.isfinite(self.entry):
            raise ValueError("entry must be positive and finite")


@dataclass(frozen=True)
class Intent:
    kind: IntentKind
    reason: str
    side: Side | None = None
    position_id: str | None = None
    fraction: float = 1.0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
