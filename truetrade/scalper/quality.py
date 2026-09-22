from __future__ import annotations

from dataclasses import dataclass
import math

from .store import ScalperStore
from .telemetry import ExecutionObservation


@dataclass(frozen=True)
class ExecutionQualitySettings:
    ewma_alpha: float = 0.08
    latency_soft_ms: float = 120.0
    latency_hard_ms: float = 750.0
    slippage_soft_spreads: float = 0.30
    slippage_hard_spreads: float = 1.50
    max_probability_penalty: float = 0.12
    min_size_multiplier: float = 0.25
    base_label_cost_spreads: float = 0.20

    def __post_init__(self) -> None:
        if not 0 < self.ewma_alpha <= 1:
            raise ValueError("invalid EWMA alpha")
        if not 0 < self.latency_soft_ms < self.latency_hard_ms:
            raise ValueError("invalid latency thresholds")
        if not 0 <= self.slippage_soft_spreads < self.slippage_hard_spreads:
            raise ValueError("invalid slippage thresholds")
        if not 0 <= self.max_probability_penalty < 0.25:
            raise ValueError("invalid probability penalty")
        if not 0 < self.min_size_multiplier <= 1:
            raise ValueError("invalid minimum size multiplier")


class ExecutionQualityController:
    META = "scalper_execution_quality_v1"

    def __init__(self, store: ScalperStore, settings: ExecutionQualitySettings | None = None):
        self.store = store
        self.settings = settings or ExecutionQualitySettings()
        saved = store.meta(self.META, {}) or {}
        self.latency_ewma_ms = float(saved.get("latency_ewma_ms", 0.0))
        self.slippage_ewma_spreads = float(saved.get("slippage_ewma_spreads", 0.0))
        self.success_ewma = float(saved.get("success_ewma", 1.0))
        self.observations = int(saved.get("observations", 0))
        self.uncertain_events = int(saved.get("uncertain_events", 0))
        self.rejections = int(saved.get("rejections", 0))

    def _ewma(self, previous: float, value: float) -> float:
        if self.observations == 0:
            return float(value)
        a = self.settings.ewma_alpha
        return (1.0 - a) * previous + a * float(value)

    def observe(self, observation: ExecutionObservation) -> None:
        self.latency_ewma_ms = self._ewma(self.latency_ewma_ms, observation.latency_ms)
        self.slippage_ewma_spreads = self._ewma(self.slippage_ewma_spreads, max(0.0, observation.slippage_spreads))
        self.success_ewma = self._ewma(self.success_ewma, 1.0 if observation.success else 0.0)
        self.observations += 1
        self._persist()

    def observe_failure(self, *, uncertain: bool) -> None:
        a = self.settings.ewma_alpha
        self.success_ewma = (1.0 - a) * self.success_ewma
        self.observations += 1
        if uncertain:
            self.uncertain_events += 1
        else:
            self.rejections += 1
        self._persist()

    @staticmethod
    def _pressure(value: float, soft: float, hard: float) -> float:
        if value <= soft:
            return 0.0
        if value >= hard:
            return 1.0
        return (value - soft) / (hard - soft)

    @property
    def pressure(self) -> float:
        cfg = self.settings
        latency = self._pressure(self.latency_ewma_ms, cfg.latency_soft_ms, cfg.latency_hard_ms)
        slip = self._pressure(self.slippage_ewma_spreads, cfg.slippage_soft_spreads, cfg.slippage_hard_spreads)
        failure = min(1.0, max(0.0, (0.98 - self.success_ewma) / 0.18))
        return min(1.0, 0.35 * latency + 0.45 * slip + 0.20 * failure)

    @property
    def size_multiplier(self) -> float:
        return max(self.settings.min_size_multiplier, 1.0 - 0.75 * self.pressure)

    @property
    def probability_penalty(self) -> float:
        return self.settings.max_probability_penalty * self.pressure

    @property
    def label_cost_spreads(self) -> float:
        return self.settings.base_label_cost_spreads + self.slippage_ewma_spreads

    @property
    def blocked(self) -> bool:
        return self.observations >= 10 and (
            self.latency_ewma_ms >= self.settings.latency_hard_ms
            or self.slippage_ewma_spreads >= self.settings.slippage_hard_spreads
            or self.success_ewma < 0.80
        )

    def snapshot(self) -> dict:
        values = {
            "latency_ewma_ms": self.latency_ewma_ms,
            "slippage_ewma_spreads": self.slippage_ewma_spreads,
            "success_ewma": self.success_ewma,
            "observations": self.observations,
            "uncertain_events": self.uncertain_events,
            "rejections": self.rejections,
            "pressure": self.pressure,
            "size_multiplier": self.size_multiplier,
            "probability_penalty": self.probability_penalty,
            "label_cost_spreads": self.label_cost_spreads,
            "blocked": self.blocked,
        }
        if not all(math.isfinite(v) for v in values.values() if isinstance(v, float)):
            raise ValueError("non-finite execution quality state")
        return values

    def _persist(self) -> None:
        self.store.set_meta(self.META, self.snapshot())
