from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .store import ScalperStore


@dataclass(frozen=True)
class SampleLabelInterval:
    feature_ts_ns: int
    label_end_ts_ns: int
    ticks_observed: int

    def __post_init__(self) -> None:
        if self.feature_ts_ns <= 0 or self.label_end_ts_ns <= self.feature_ts_ns:
            raise ValueError("invalid sample label interval")
        if self.ticks_observed < 1:
            raise ValueError("ticks_observed must be positive")


def ensure_label_interval_table(store: ScalperStore) -> None:
    with store.db:
        store.db.execute(
            """
            CREATE TABLE IF NOT EXISTS sample_label_intervals(
                feature_ts_ns INTEGER PRIMARY KEY,
                label_end_ts_ns INTEGER NOT NULL,
                ticks_observed INTEGER NOT NULL CHECK(ticks_observed > 0)
            )
            """
        )
        store.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sample_label_intervals_end ON sample_label_intervals(label_end_ts_ns)"
        )


def clear_label_intervals(store: ScalperStore) -> None:
    ensure_label_interval_table(store)
    with store.db:
        store.db.execute("DELETE FROM sample_label_intervals")


def record_label_interval(
    store: ScalperStore,
    feature_ts_ns: int,
    label_end_ts_ns: int,
    ticks_observed: int,
) -> None:
    record_label_intervals(
        store,
        [SampleLabelInterval(int(feature_ts_ns), int(label_end_ts_ns), int(ticks_observed))],
    )


def record_label_intervals(store: ScalperStore, intervals: Iterable[SampleLabelInterval]) -> int:
    rows = list(intervals)
    if not rows:
        return 0
    ensure_label_interval_table(store)
    payload = [(r.feature_ts_ns, r.label_end_ts_ns, r.ticks_observed) for r in rows]
    with store.db:
        store.db.executemany(
            """INSERT INTO sample_label_intervals(feature_ts_ns,label_end_ts_ns,ticks_observed)
               VALUES(?,?,?)
               ON CONFLICT(feature_ts_ns) DO UPDATE SET
                   label_end_ts_ns=excluded.label_end_ts_ns,
                   ticks_observed=excluded.ticks_observed""",
            payload,
        )
    return len(payload)


def load_label_intervals(store: ScalperStore) -> dict[int, SampleLabelInterval]:
    ensure_label_interval_table(store)
    rows = store.db.execute(
        "SELECT feature_ts_ns,label_end_ts_ns,ticks_observed FROM sample_label_intervals ORDER BY feature_ts_ns"
    ).fetchall()
    return {
        int(feature_ts): SampleLabelInterval(int(feature_ts), int(label_end), int(observed))
        for feature_ts, label_end, observed in rows
    }
