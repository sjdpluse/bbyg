from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import time

from truetrade.scalper.execution import DemoMT5Settings
from truetrade.scalper.timebase import BrokerTimebase, TimeNormalizedDemoMT5Execution


def _dt_from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _describe(rows, timebase: BrokerTimebase) -> dict:
    if rows is None or len(rows) == 0:
        return {"count": 0}
    first_ms = int(rows[0]["time_msc"])
    last_ms = int(rows[-1]["time_msc"])
    first_norm = datetime.fromtimestamp(timebase.msc_to_utc_ns(first_ms) / 1_000_000_000, tz=timezone.utc)
    last_norm = datetime.fromtimestamp(timebase.msc_to_utc_ns(last_ms) / 1_000_000_000, tz=timezone.utc)
    return {
        "count": int(len(rows)),
        "first_raw_as_utc": _iso(_dt_from_ms(first_ms)),
        "last_raw_as_utc": _iso(_dt_from_ms(last_ms)),
        "first_normalized_utc": _iso(first_norm),
        "last_normalized_utc": _iso(last_norm),
    }


def main() -> None:
    broker = TimeNormalizedDemoMT5Execution(DemoMT5Settings.from_env())
    timebase = BrokerTimebase.from_env()
    try:
        broker.connect()
        raw_latest = broker.call("symbol_info_tick", broker.symbol)
        raw_ms = int(raw_latest.time_msc)
        raw_as_utc = _dt_from_ms(raw_ms)
        normalized_utc = datetime.fromtimestamp(
            timebase.msc_to_utc_ns(raw_ms) / 1_000_000_000,
            tz=timezone.utc,
        )
        system_utc = datetime.fromtimestamp(time.time(), tz=timezone.utc)

        raw_start = raw_as_utc - timedelta(minutes=5)
        normalized_start = normalized_utc - timedelta(minutes=5)
        raw_rows = broker.call(
            "copy_ticks_range", broker.symbol, raw_start, raw_as_utc, broker.api.COPY_TICKS_ALL
        )
        normalized_rows = broker.call(
            "copy_ticks_range", broker.symbol, normalized_start, normalized_utc, broker.api.COPY_TICKS_ALL
        )

        print(json.dumps({
            "mode": "read_only_mt5_time_probe",
            "symbol": broker.symbol,
            "configured_offset_seconds": timebase.utc_offset_seconds,
            "system_utc": _iso(system_utc),
            "latest_tick_raw_as_utc": _iso(raw_as_utc),
            "latest_tick_normalized_utc": _iso(normalized_utc),
            "raw_minus_system_seconds": (raw_as_utc - system_utc).total_seconds(),
            "normalized_minus_system_seconds": (normalized_utc - system_utc).total_seconds(),
            "queries": {
                "raw_window": {
                    "requested_start_utc": _iso(raw_start),
                    "requested_end_utc": _iso(raw_as_utc),
                    "result": _describe(raw_rows, timebase),
                },
                "normalized_window": {
                    "requested_start_utc": _iso(normalized_start),
                    "requested_end_utc": _iso(normalized_utc),
                    "result": _describe(normalized_rows, timebase),
                },
            },
        }, sort_keys=True), flush=True)
    finally:
        broker.shutdown()


if __name__ == "__main__":
    main()
