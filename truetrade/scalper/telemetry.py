from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .types import Side


@dataclass(frozen=True)
class ExecutionObservation:
    latency_ms: float
    slippage_price: float
    slippage_spreads: float
    size: float
    success: bool

    def __post_init__(self) -> None:
        if self.latency_ms < 0 or self.size <= 0:
            raise ValueError("invalid execution observation")
        if not all(math.isfinite(v) for v in (
            self.latency_ms, self.slippage_price, self.slippage_spreads, self.size
        )):
            raise ValueError("execution telemetry must be finite")


class ExecutionTelemetry:
    def __init__(self, max_samples: int = 5000):
        if max_samples < 10:
            raise ValueError("max_samples too small")
        self.max_samples = max_samples
        self.samples: list[ExecutionObservation] = []

    def observe(self, observation: ExecutionObservation) -> None:
        self.samples.append(observation)
        if len(self.samples) > self.max_samples:
            del self.samples[:len(self.samples) - self.max_samples]

    def record(self, *, start_ns: int, end_ns: int, expected_price: float, fill_price: float,
               spread: float, side: Side, size: float, success: bool = True) -> ExecutionObservation:
        if end_ns < start_ns or expected_price <= 0 or fill_price <= 0 or spread <= 0:
            raise ValueError("invalid execution measurement")
        slip = (fill_price - expected_price) * side.sign
        obs = ExecutionObservation((end_ns - start_ns) / 1e6, slip, slip / spread, size, success)
        self.observe(obs)
        return obs

    @staticmethod
    def _percentile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = (len(ordered) - 1) * q
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        weight = index - low
        return ordered[low] * (1 - weight) + ordered[high] * weight

    def summary(self) -> dict:
        if not self.samples:
            return {"count": 0, "success_rate": None, "latency_p50_ms": None,
                    "latency_p95_ms": None, "slippage_p50_spreads": None,
                    "slippage_p95_spreads": None}
        lat = [s.latency_ms for s in self.samples]
        slip = [s.slippage_spreads for s in self.samples]
        return {
            "count": len(self.samples),
            "success_rate": sum(s.success for s in self.samples) / len(self.samples),
            "latency_p50_ms": self._percentile(lat, 0.50),
            "latency_p95_ms": self._percentile(lat, 0.95),
            "slippage_p50_spreads": self._percentile(slip, 0.50),
            "slippage_p95_spreads": self._percentile(slip, 0.95),
        }

    @staticmethod
    def serialize(observation: ExecutionObservation) -> dict:
        return asdict(observation)

    @staticmethod
    def deserialize(document: dict) -> ExecutionObservation:
        return ExecutionObservation(
            float(document["latency_ms"]), float(document["slippage_price"]),
            float(document["slippage_spreads"]), float(document["size"]),
            bool(document["success"]),
        )
