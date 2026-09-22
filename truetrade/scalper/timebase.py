from __future__ import annotations

from dataclasses import dataclass
import os

from .execution import DemoMT5Execution
from .types import PositionState, Tick


@dataclass(frozen=True)
class BrokerTimebase:
    """Translate broker-encoded epoch values onto UTC.

    Some MT5 servers expose ``time_msc`` values whose epoch is shifted by the
    server UTC offset. BBYG stores one canonical UTC nanosecond timebase so
    live ticks, historical ticks, position times and deal outcomes can be
    compared safely.
    """

    utc_offset_seconds: int = 0

    @classmethod
    def from_env(cls) -> "BrokerTimebase":
        return cls(int(os.getenv("MT5_SERVER_UTC_OFFSET_SECONDS", "0")))

    @property
    def offset_ns(self) -> int:
        return int(self.utc_offset_seconds) * 1_000_000_000

    def msc_to_utc_ns(self, time_msc: int) -> int:
        return int(time_msc) * 1_000_000 - self.offset_ns

    def ns_to_utc_ns(self, timestamp_ns: int) -> int:
        return int(timestamp_ns) - self.offset_ns


class TimeNormalizedDemoMT5Execution(DemoMT5Execution):
    """DEMO adapter that exposes all runtime timestamps on a UTC timebase."""

    def __init__(self, settings, api=None, *, timebase: BrokerTimebase | None = None):
        super().__init__(settings, api=api)
        self.timebase = timebase or BrokerTimebase.from_env()

    def latest_tick(self) -> Tick | None:
        self._account()
        raw = self.call("symbol_info_tick", self.symbol)
        last = float(getattr(raw, "last", 0.0) or 0.0)
        volume = float(getattr(raw, "volume_real", getattr(raw, "volume", 0.0)) or 0.0)
        signature = (int(raw.time_msc), float(raw.bid), float(raw.ask), last, volume)
        if signature == self._last_tick_signature:
            return None
        self._last_tick_signature = signature
        raw_ns = self.timebase.msc_to_utc_ns(int(raw.time_msc))
        ts_ns = max(raw_ns, self._last_tick_ns + 1)
        self._last_tick_ns = ts_ns
        return Tick(ts_ns, float(raw.bid), float(raw.ask), last, volume)

    def _fresh_tick_for_write(self) -> Tick:
        tick = self.latest_tick()
        if tick is not None:
            return tick
        raw = self.call("symbol_info_tick", self.symbol)
        raw_ns = self.timebase.msc_to_utc_ns(int(raw.time_msc))
        return Tick(
            max(raw_ns, self._last_tick_ns + 1),
            float(raw.bid),
            float(raw.ask),
            float(getattr(raw, "last", 0.0) or 0.0),
            float(getattr(raw, "volume_real", getattr(raw, "volume", 0.0)) or 0.0),
        )

    def positions(self) -> list[PositionState]:
        positions = super().positions()
        if not self.timebase.utc_offset_seconds:
            return positions
        for p in positions:
            p.opened_ns = self.timebase.ns_to_utc_ns(p.opened_ns)
        return positions

    def closed_outcome(self, position_identifier: int) -> dict | None:
        outcome = super().closed_outcome(position_identifier)
        if outcome is None or not self.timebase.utc_offset_seconds:
            return outcome
        outcome = dict(outcome)
        outcome["opened_ns"] = self.timebase.ns_to_utc_ns(int(outcome["opened_ns"]))
        outcome["closed_ns"] = self.timebase.ns_to_utc_ns(int(outcome["closed_ns"]))
        return outcome
