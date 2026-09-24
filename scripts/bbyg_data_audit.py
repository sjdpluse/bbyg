from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path

from truetrade.scalper.store import ScalperStore


def _iso(ns: int | None) -> str | None:
    if ns is None:
        return None
    return datetime.fromtimestamp(int(ns) / 1_000_000_000, tz=timezone.utc).isoformat()


def main() -> None:
    state_dir = Path(os.getenv("BBYG_STATE_DIR", "data/bbyg-demo"))
    store = ScalperStore(state_dir / "scalper.sqlite")
    try:
        db = store.db
        count, first_ns, last_ns = db.execute(
            "SELECT count(*), min(ts_ns), max(ts_ns) FROM ticks"
        ).fetchone()
        count = int(count or 0)
        if count == 0:
            raise SystemExit("no stored ticks")
        first_ns = int(first_ns)
        last_ns = int(last_ns)

        per_date = [
            {"utc_date": str(day), "ticks": int(n)}
            for day, n in db.execute(
                """SELECT strftime('%Y-%m-%d', ts_ns / 1000000000, 'unixepoch') AS d, count(*)
                   FROM ticks GROUP BY d ORDER BY d"""
            ).fetchall()
        ]
        per_hour = [
            {"utc_hour": str(hour), "ticks": int(n)}
            for hour, n in db.execute(
                """SELECT strftime('%Y-%m-%dT%H:00Z', ts_ns / 1000000000, 'unixepoch') AS h, count(*)
                   FROM ticks GROUP BY h ORDER BY h"""
            ).fetchall()
        ]
        repeated_ms = db.execute(
            """SELECT count(*), coalesce(sum(n),0), coalesce(max(n),0)
               FROM (
                 SELECT CAST(ts_ns / 1000000 AS INTEGER) AS ms, count(*) AS n
                 FROM ticks GROUP BY ms HAVING count(*) > 1
               )"""
        ).fetchone()
        repeated_ms_count = int(repeated_ms[0] or 0)
        ticks_in_repeated_ms = int(repeated_ms[1] or 0)
        max_same_ms_cluster = int(repeated_ms[2] or 0)

        exact_quote_repeats = int(db.execute(
            """SELECT coalesce(sum(n - 1),0)
               FROM (
                 SELECT CAST(ts_ns / 1000000 AS INTEGER) AS ms, bid, ask, last, volume, count(*) AS n
                 FROM ticks
                 GROUP BY ms,bid,ask,last,volume
                 HAVING count(*) > 1
               )"""
        ).fetchone()[0] or 0)

        sample_count = store.sample_count()
        interval_count = int(db.execute(
            """SELECT count(*) FROM sqlite_master WHERE type='table' AND name='sample_label_intervals'"""
        ).fetchone()[0])
        label_intervals = 0
        if interval_count:
            label_intervals = int(db.execute("SELECT count(*) FROM sample_label_intervals").fetchone()[0])

        span_seconds = max(0.0, (last_ns - first_ns) / 1_000_000_000)
        print(json.dumps({
            "mode": "read_only_data_audit",
            "state_dir": str(state_dir),
            "stored_ticks": count,
            "first_tick_utc": _iso(first_ns),
            "last_tick_utc": _iso(last_ns),
            "span_hours": span_seconds / 3600.0,
            "utc_dates": len(per_date),
            "utc_hours": len(per_hour),
            "per_date": per_date,
            "per_hour": per_hour,
            "same_millisecond": {
                "milliseconds_with_multiple_ticks": repeated_ms_count,
                "ticks_in_multi_tick_milliseconds": ticks_in_repeated_ms,
                "fraction_of_ticks_in_multi_tick_milliseconds": ticks_in_repeated_ms / count,
                "max_ticks_in_one_millisecond": max_same_ms_cluster,
                "exact_same_ms_quote_repeats_beyond_first": exact_quote_repeats,
            },
            "samples": sample_count,
            "label_intervals": label_intervals,
            "sample_to_tick_ratio": sample_count / count,
        }, sort_keys=True), flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    main()
