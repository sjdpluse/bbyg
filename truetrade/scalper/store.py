from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Iterable

from .learning import Sample
from .types import PositionState, Side, Tick


@dataclass(frozen=True)
class StoredSample:
    sample_id: int
    feature_ts_ns: int
    sample: Sample


class ScalperStore:
    """Durable local state for ticks, learning samples, positions, and execution events."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS ticks(
                ts_ns INTEGER PRIMARY KEY,
                bid REAL NOT NULL,
                ask REAL NOT NULL,
                last REAL NOT NULL,
                volume REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS samples(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feature_ts_ns INTEGER NOT NULL UNIQUE,
                x_json TEXT NOT NULL,
                y INTEGER NOT NULL CHECK(y IN (0,1))
            );
            CREATE TABLE IF NOT EXISTS positions(
                position_id TEXT PRIMARY KEY,
                side TEXT NOT NULL CHECK(side IN ('LONG','SHORT')),
                size REAL NOT NULL,
                entry REAL NOT NULL,
                opened_ns INTEGER NOT NULL,
                peak_exit_price REAL,
                trough_exit_price REAL,
                reductions INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ns INTEGER NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS execution_intents(
                decision_id TEXT PRIMARY KEY,
                created_ns INTEGER NOT NULL,
                kind TEXT NOT NULL,
                position_id TEXT,
                side TEXT,
                size REAL,
                fraction REAL NOT NULL,
                state TEXT NOT NULL,
                broker_position_id TEXT,
                detail TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "ScalperStore":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def append_tick(self, tick: Tick) -> bool:
        with self.db:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO ticks(ts_ns,bid,ask,last,volume) VALUES(?,?,?,?,?)",
                (tick.ts_ns, tick.bid, tick.ask, tick.last, tick.volume),
            )
        return cur.rowcount == 1

    def ticks(self, *, after_ns: int = 0, limit: int | None = None) -> list[Tick]:
        sql = "SELECT ts_ns,bid,ask,last,volume FROM ticks WHERE ts_ns>? ORDER BY ts_ns"
        args: list[object] = [int(after_ns)]
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            sql += " LIMIT ?"
            args.append(int(limit))
        return [Tick(*row) for row in self.db.execute(sql, args).fetchall()]

    def last_tick_ns(self) -> int:
        row = self.db.execute("SELECT max(ts_ns) FROM ticks").fetchone()
        return int(row[0] or 0)

    def add_sample(self, feature_ts_ns: int, sample: Sample) -> bool:
        payload = json.dumps(list(sample.x), separators=(",", ":"), allow_nan=False)
        with self.db:
            cur = self.db.execute(
                "INSERT OR IGNORE INTO samples(feature_ts_ns,x_json,y) VALUES(?,?,?)",
                (int(feature_ts_ns), payload, int(sample.y)),
            )
        return cur.rowcount == 1

    def samples(self, *, after_id: int = 0) -> list[StoredSample]:
        rows = self.db.execute(
            "SELECT id,feature_ts_ns,x_json,y FROM samples WHERE id>? ORDER BY id", (int(after_id),)
        ).fetchall()
        return [StoredSample(int(i), int(ts), Sample(tuple(json.loads(x)), int(y))) for i, ts, x, y in rows]

    def sample_count(self) -> int:
        return int(self.db.execute("SELECT count(*) FROM samples").fetchone()[0])

    def replace_positions(self, positions: Iterable[PositionState]) -> None:
        rows = list(positions)
        with self.db:
            self.db.execute("DELETE FROM positions")
            self.db.executemany(
                """INSERT INTO positions(position_id,side,size,entry,opened_ns,peak_exit_price,trough_exit_price,reductions)
                   VALUES(?,?,?,?,?,?,?,?)""",
                [
                    (
                        p.position_id,
                        p.side.value,
                        p.size,
                        p.entry,
                        p.opened_ns,
                        p.peak_exit_price,
                        p.trough_exit_price,
                        p.reductions,
                    )
                    for p in rows
                ],
            )

    def load_positions(self) -> list[PositionState]:
        rows = self.db.execute(
            "SELECT position_id,side,size,entry,opened_ns,peak_exit_price,trough_exit_price,reductions FROM positions ORDER BY opened_ns,position_id"
        ).fetchall()
        return [
            PositionState(str(pid), Side(side), float(size), float(entry), int(opened), peak, trough, int(reductions))
            for pid, side, size, entry, opened, peak, trough, reductions in rows
        ]

    def append_event(self, ts_ns: int, kind: str, payload: dict) -> None:
        if not kind:
            raise ValueError("event kind required")
        document = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self.db:
            self.db.execute(
                "INSERT INTO events(ts_ns,kind,payload_json) VALUES(?,?,?)", (int(ts_ns), kind, document)
            )

    def events(self, kind: str | None = None) -> list[dict]:
        if kind is None:
            rows = self.db.execute("SELECT ts_ns,kind,payload_json FROM events ORDER BY id").fetchall()
        else:
            rows = self.db.execute(
                "SELECT ts_ns,kind,payload_json FROM events WHERE kind=? ORDER BY id", (kind,)
            ).fetchall()
        return [{"ts_ns": int(ts), "kind": k, "payload": json.loads(p)} for ts, k, p in rows]

    def create_execution_intent(self, decision_id: str, created_ns: int, *, kind: str,
                                position_id: str | None, side: str | None, size: float | None,
                                fraction: float, detail: str) -> bool:
        if not decision_id:
            raise ValueError("decision_id required")
        with self.db:
            cur = self.db.execute(
                """INSERT OR IGNORE INTO execution_intents
                   (decision_id,created_ns,kind,position_id,side,size,fraction,state,broker_position_id,detail)
                   VALUES(?,?,?,?,?,?,?,'created',NULL,?)""",
                (decision_id, int(created_ns), kind, position_id, side, size, float(fraction), detail),
            )
        return cur.rowcount == 1

    def transition_execution_intent(self, decision_id: str, state: str, *,
                                    broker_position_id: str | None = None, detail: str | None = None) -> None:
        allowed = {"created", "submitted", "confirmed", "rejected", "unknown"}
        if state not in allowed:
            raise ValueError("invalid execution intent state")
        row = self.db.execute(
            "SELECT state,detail FROM execution_intents WHERE decision_id=?", (decision_id,)
        ).fetchone()
        if row is None:
            raise KeyError(decision_id)
        current, current_detail = row
        transitions = {
            "created": {"submitted", "rejected", "unknown"},
            "submitted": {"confirmed", "rejected", "unknown"},
            "unknown": {"confirmed", "rejected", "unknown"},
            "confirmed": set(),
            "rejected": set(),
        }
        if state != current and state not in transitions[current]:
            raise ValueError(f"invalid execution transition {current}->{state}")
        with self.db:
            self.db.execute(
                """UPDATE execution_intents
                   SET state=?, broker_position_id=COALESCE(?,broker_position_id), detail=?
                   WHERE decision_id=?""",
                (state, broker_position_id, current_detail if detail is None else detail, decision_id),
            )

    def unsettled_execution_intents(self) -> list[dict]:
        rows = self.db.execute(
            """SELECT decision_id,created_ns,kind,position_id,side,size,fraction,state,broker_position_id,detail
               FROM execution_intents WHERE state IN ('created','submitted','unknown') ORDER BY created_ns"""
        ).fetchall()
        keys = ("decision_id","created_ns","kind","position_id","side","size","fraction","state","broker_position_id","detail")
        return [dict(zip(keys, row)) for row in rows]

    def execution_intent(self, decision_id: str) -> dict | None:
        row = self.db.execute(
            """SELECT decision_id,created_ns,kind,position_id,side,size,fraction,state,broker_position_id,detail
               FROM execution_intents WHERE decision_id=?""", (decision_id,)
        ).fetchone()
        if row is None:
            return None
        keys = ("decision_id","created_ns","kind","position_id","side","size","fraction","state","broker_position_id","detail")
        return dict(zip(keys, row))

    def set_meta(self, key: str, value: str | int | float | bool | dict) -> None:
        if not key:
            raise ValueError("meta key required")
        encoded = json.dumps(value, separators=(",", ":"), allow_nan=False)
        with self.db:
            self.db.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, encoded),
            )

    def meta(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])
