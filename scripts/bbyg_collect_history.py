from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time

from truetrade.scalper.execution import DemoMT5Settings
from truetrade.scalper.history_import import insert_tick_payload, normalize_history_rows
from truetrade.scalper.store import ScalperStore
from truetrade.scalper.timebase import BrokerTimebase, TimeNormalizedDemoMT5Execution


def _print(stage: str, **fields) -> None:
    print(json.dumps({"stage": stage, **fields}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect multi-day MT5 DEMO history into BBYG in safe chunks")
    parser.add_argument("--days", type=float, default=7.0, help="calendar days of broker history to request")
    parser.add_argument("--chunk-hours", type=float, default=3.0, help="request window size; default 3h")
    parser.add_argument("--max-total-ticks", type=int, default=3_000_000, help="hard safety cap")
    args = parser.parse_args()
    if not 0 < args.days <= 30:
        raise SystemExit("--days must be in (0, 30]")
    if not 0.25 <= args.chunk_hours <= 12:
        raise SystemExit("--chunk-hours must be between 0.25 and 12")
    if not 100_000 <= args.max_total_ticks <= 10_000_000:
        raise SystemExit("--max-total-ticks must be between 100,000 and 10,000,000")

    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-multiday"))
    state_dir.mkdir(parents=True, exist_ok=True)
    store = ScalperStore(state_dir / "scalper.sqlite")
    broker = TimeNormalizedDemoMT5Execution(DemoMT5Settings.from_env())
    timebase = BrokerTimebase.from_env()
    started = time.perf_counter()

    try:
        broker.connect()
        latest = broker.call("symbol_info_tick", broker.symbol)
        raw_end = datetime.fromtimestamp(int(latest.time_msc) / 1000.0, tz=timezone.utc)
        raw_start = raw_end - timedelta(days=args.days)
        step = timedelta(hours=args.chunk_hours)
        _print(
            "collect_start",
            symbol=broker.symbol,
            requested_days=args.days,
            chunk_hours=args.chunk_hours,
            configured_offset_seconds=timebase.utc_offset_seconds,
            raw_start=raw_start.isoformat(),
            raw_end=raw_end.isoformat(),
        )

        cursor = raw_start
        total_inserted = 0
        total_received = 0
        total_invalid = 0
        chunks = 0
        nonempty_chunks = 0
        first_normalized_ns = None
        last_normalized_ns = None

        while cursor < raw_end:
            end = min(cursor + step, raw_end)
            rows = broker.call("copy_ticks_range", broker.symbol, cursor, end, broker.api.COPY_TICKS_ALL)
            chunks += 1
            received = 0 if rows is None else len(rows)
            total_received += received
            inserted = 0
            invalid = 0
            first_raw = None
            last_raw = None
            if received:
                nonempty_chunks += 1
                batch = normalize_history_rows(rows, timebase)
                invalid = batch.invalid_rows
                first_raw = batch.first_raw_msc
                last_raw = batch.last_raw_msc
                total_invalid += invalid
                if batch.payload:
                    inserted = insert_tick_payload(store, batch.payload)
                    total_inserted += inserted
                    if first_normalized_ns is None:
                        first_normalized_ns = batch.payload[0][0]
                    last_normalized_ns = batch.payload[-1][0]

            _print(
                "chunk",
                chunk=chunks,
                requested_start=cursor.isoformat(),
                requested_end=end.isoformat(),
                received=received,
                inserted=inserted,
                invalid=invalid,
                first_raw_msc=first_raw,
                last_raw_msc=last_raw,
                total_inserted=total_inserted,
            )
            if total_inserted >= args.max_total_ticks:
                raise SystemExit(f"hard tick cap reached at {total_inserted}; increase --max-total-ticks deliberately")

            # Make broker request windows non-overlapping at millisecond precision.
            cursor = end + timedelta(milliseconds=1)

        first_iso = None
        last_iso = None
        if first_normalized_ns is not None:
            first_iso = datetime.fromtimestamp(first_normalized_ns / 1_000_000_000, tz=timezone.utc).isoformat()
        if last_normalized_ns is not None:
            last_iso = datetime.fromtimestamp(last_normalized_ns / 1_000_000_000, tz=timezone.utc).isoformat()

        _print(
            "complete",
            symbol=broker.symbol,
            chunks=chunks,
            nonempty_chunks=nonempty_chunks,
            total_received=total_received,
            total_inserted=total_inserted,
            total_invalid=total_invalid,
            stored_ticks=int(store.db.execute("SELECT count(*) FROM ticks").fetchone()[0]),
            first_normalized_utc=first_iso,
            last_normalized_utc=last_iso,
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
    finally:
        broker.shutdown()
        store.close()


if __name__ == "__main__":
    main()
