from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .types import Side


class ThesisState(str, Enum):
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    ENTERED = "ENTERED"
    DEVELOPING = "DEVELOPING"
    PROTECTED = "PROTECTED"
    EXITING = "EXITING"
    CLOSED = "CLOSED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class ThesisSettings:
    min_entry_observations: int = 6
    min_entry_age_seconds: float = 2.0
    min_entry_confidence: float = 0.60
    min_memory_confidence: float = 0.20
    max_novelty: float = 0.85
    min_directional_margin: float = 0.08
    max_missed_observations: int = 2
    min_exit_observations: int = 4
    min_exit_age_seconds: float = 15.0

    def __post_init__(self) -> None:
        if self.min_entry_observations < 2 or self.min_exit_observations < 2:
            raise ValueError("thesis confirmation windows must be >=2")
        if self.min_entry_age_seconds < 0 or self.min_exit_age_seconds < 0:
            raise ValueError("thesis ages must be non-negative")
        if not 0.5 < self.min_entry_confidence < 1:
            raise ValueError("invalid entry confidence")
        if not 0 <= self.min_memory_confidence <= 1:
            raise ValueError("invalid memory confidence")
        if not 0 <= self.max_novelty <= 1:
            raise ValueError("invalid novelty threshold")
        if not 0 <= self.min_directional_margin < 1:
            raise ValueError("invalid directional margin")
        if self.max_missed_observations < 0:
            raise ValueError("invalid miss allowance")


@dataclass(frozen=True)
class ThesisEvidence:
    ts_ns: int
    side: Side
    model_confidence: float
    technical_score: int
    technical_opposite_score: int
    trend_efficiency: float
    velocity: float
    memory_confidence: float
    memory_directional_bias: float | None
    novelty: float
    regime_ok: bool
    fundamental_ok: bool

    def __post_init__(self) -> None:
        if self.ts_ns <= 0:
            raise ValueError("evidence timestamp must be positive")
        if not 0 <= self.model_confidence <= 1:
            raise ValueError("invalid model confidence")
        if not 0 <= self.memory_confidence <= 1 or not 0 <= self.novelty <= 1:
            raise ValueError("invalid memory evidence")
        values = (self.trend_efficiency, self.velocity)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("non-finite market evidence")
        if self.memory_directional_bias is not None and not math.isfinite(self.memory_directional_bias):
            raise ValueError("invalid memory bias")


@dataclass
class TradeThesis:
    thesis_id: str
    side: Side
    created_ns: int
    settings: ThesisSettings
    state: ThesisState = ThesisState.CANDIDATE
    evidence_count: int = 0
    consecutive_support: int = 0
    consecutive_exit: int = 0
    misses: int = 0
    last_evidence_ns: int = 0
    entered_ns: int | None = None
    reject_reason: str | None = None

    def _entry_support(self, evidence: ThesisEvidence) -> bool:
        if evidence.side is not self.side:
            return False
        if evidence.model_confidence < self.settings.min_entry_confidence:
            return False
        if not evidence.regime_ok or not evidence.fundamental_ok:
            return False
        if evidence.technical_score <= evidence.technical_opposite_score:
            return False
        if evidence.trend_efficiency * self.side.sign <= 0:
            return False
        if evidence.velocity * self.side.sign <= 0:
            return False
        if evidence.novelty > self.settings.max_novelty:
            return False
        if evidence.memory_confidence >= self.settings.min_memory_confidence:
            bias = evidence.memory_directional_bias
            if bias is not None and bias * self.side.sign < self.settings.min_directional_margin:
                return False
        return True

    def observe_entry(self, evidence: ThesisEvidence) -> ThesisState:
        if self.state not in {ThesisState.CANDIDATE, ThesisState.CONFIRMED}:
            return self.state
        self.evidence_count += 1
        self.last_evidence_ns = evidence.ts_ns
        if self._entry_support(evidence):
            self.consecutive_support += 1
            self.misses = 0
        else:
            self.misses += 1
            if self.misses > self.settings.max_missed_observations:
                self.state = ThesisState.REJECTED
                self.reject_reason = "entry_evidence_decayed"
                return self.state
            self.consecutive_support = max(0, self.consecutive_support - 1)

        age = (evidence.ts_ns - self.created_ns) / 1_000_000_000.0
        if (
            self.consecutive_support >= self.settings.min_entry_observations
            and age >= self.settings.min_entry_age_seconds
        ):
            self.state = ThesisState.CONFIRMED
        return self.state

    def mark_entered(self, ts_ns: int) -> None:
        if self.state is not ThesisState.CONFIRMED:
            raise ValueError("only confirmed thesis may enter")
        if ts_ns < self.created_ns:
            raise ValueError("entry timestamp precedes thesis")
        self.state = ThesisState.ENTERED
        self.entered_ns = int(ts_ns)
        self.consecutive_exit = 0

    def mark_developing(self) -> None:
        if self.state in {ThesisState.ENTERED, ThesisState.PROTECTED}:
            self.state = ThesisState.DEVELOPING

    def mark_protected(self) -> None:
        if self.state in {ThesisState.ENTERED, ThesisState.DEVELOPING}:
            self.state = ThesisState.PROTECTED

    def observe_exit(self, evidence: ThesisEvidence) -> ThesisState:
        if self.state not in {ThesisState.ENTERED, ThesisState.DEVELOPING, ThesisState.PROTECTED}:
            return self.state
        if self.entered_ns is None:
            raise ValueError("entered thesis missing timestamp")
        age = (evidence.ts_ns - self.entered_ns) / 1_000_000_000.0
        opposite = evidence.side is not self.side
        technical_reversal = evidence.technical_opposite_score > evidence.technical_score
        memory_reversal = (
            evidence.memory_confidence >= self.settings.min_memory_confidence
            and evidence.memory_directional_bias is not None
            and evidence.memory_directional_bias * self.side.sign < -self.settings.min_directional_margin
        )
        exit_support = opposite and (technical_reversal or memory_reversal)
        if exit_support:
            self.consecutive_exit += 1
        else:
            self.consecutive_exit = 0
        if age >= self.settings.min_exit_age_seconds and self.consecutive_exit >= self.settings.min_exit_observations:
            self.state = ThesisState.EXITING
        return self.state

    def close(self) -> None:
        if self.state not in {
            ThesisState.ENTERED,
            ThesisState.DEVELOPING,
            ThesisState.PROTECTED,
            ThesisState.EXITING,
        }:
            raise ValueError("invalid thesis close")
        self.state = ThesisState.CLOSED
