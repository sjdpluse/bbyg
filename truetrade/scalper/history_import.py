from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .store import ScalperStore
from .timebase import BrokerTimebase


@dataclass(frozen=True)
class HistoryBatch:
    payload: tuple[tuple[int, float, float, float, float], ...]
    invalid_rows: int
    first_raw_msc: int | None
    last_raw_msc: int | None


def _volume(row) -> float:
    names = getattr(getattr(row, "dtype", None), "names", None) or ()
    if "volume_real" in names:
        return float(row["volume_real"])
    try:
        return float(row["volume_real"])
    except (KeyError, TypeError, ValueError, IndexError):
        return float(row["volume"])


def normalize_history_rows(rows: Iterable, timebase: BrokerTimebase) -> HistoryBatch:
    """Convert MT5 history rows to deterministic canonical UTC-ns tick rows.

    MT5 can emit multiple legitimate quotes with the same millisecond timestamp. They are
    preserved in source order by allocating +1ns, +2ns, ... within that millisecond only.
    Chunk windows must therefore be non-overlapping at millisecond precision.
    """
    payload: list[tuple[int, float, float, float, float]] = []
    invalid = 0
    first_raw: int | None = None
    last_raw: int | None = None
    current_msc: int | None = None
    sequence = 0

    for row in rows:
        raw_msc = int(row["time_msc"])
        bid = float(row["bid"])
        ask = float(row["ask"])
        if raw_msc <= 0 or bid <= 0 or ask <= 0 or ask < bid:
            invalid += 1
            continue
        last = float(row["last"])
        volume = _volume(row)
        if current_msc == raw_msc:
            sequence += 1
        else:
            current_msc = raw_msc
            sequence = 0
        ts_ns = timebase.msc_to_utc_ns(raw_msc) + sequence
        payload.append((ts_ns, bid, ask, last, volume))
        if first_raw is None:
            first_raw = raw_msc
        last_raw = raw_msc

    return HistoryBatch(tuple(payload), invalid, first_raw, last_raw)


def insert_tick_payload(store: ScalperStore, payload: Iterable[tuple[int, float, float, float, float]], *, batch_size: int = 25_000) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = list(payload)
    inserted = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        before = store.db.total_changes
        with store.db:
            store.db.executemany(
                "INSERT OR IGNORE INTO ticks(ts_ns,bid,ask,last,volume) VALUES(?,?,?,?,?)",
                batch,
            )
        inserted += int(store.db.total_changes - before)
    return inserted
