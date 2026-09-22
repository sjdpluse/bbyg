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
    """Durable local state for ticks, learning samples, positions, execution, and forward evidence."""

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
                reductions INTEGER NOT NULL DEFAULT 0,
                broker_stop REAL
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
                stop REAL,
                state TEXT NOT NULL,
                broker_position_id TEXT,
                broker_identifier INTEGER,
                model_generation INTEGER NOT NULL DEFAULT 0,
                entry_equity REAL,
                risk_amount REAL,
                detail TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trade_outcomes(
                position_id TEXT PRIMARY KEY,
                decision_id TEXT NOT NULL UNIQUE,
                broker_identifier INTEGER NOT NULL,
                model_generation INTEGER NOT NULL,
                opened_ns INTEGER NOT NULL,
                closed_ns INTEGER NOT NULL,
                net_pnl REAL NOT NULL,
                profit REAL NOT NULL,
                commission REAL NOT NULL,
                swap REAL NOT NULL,
                fee REAL NOT NULL,
                volume REAL NOT NULL,
                risk_amount REAL NOT NULL,
                entry_equity REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS meta(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self._ensure_column("positions", "broker_stop", "REAL")
        self._ensure_column("execution_intents", "stop", "REAL")
        self._ensure_column("execution_intents", "broker_identifier", "INTEGER")
        self._ensure_column("execution_intents", "model_generation", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("execution_intents", "entry_equity", "REAL")
        self._ensure_column("execution_intents", "risk_amount", "REAL")
        self.db.commit()

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        names = {str(row[1]) for row in self.db.execute(f"PRAGMA table_info({table})")}
        if column not in names:
            self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

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
        return self.add_samples([(int(feature_ts_ns), sample)]) == 1

    def add_samples(self, rows: Iterable[tuple[int, Sample]]) -> int:
        payload = [
            (
                int(feature_ts_ns),
                json.dumps(list(sample.x), separators=(",", ":"), allow_nan=False),
                int(sample.y),
            )
            for feature_ts_ns, sample in rows
        ]
        if not payload:
            return 0
        before = self.db.total_changes
        with self.db:
            self.db.executemany(
                "INSERT OR IGNORE INTO samples(feature_ts_ns,x_json,y) VALUES(?,?,?)",
                payload,
            )
        return int(self.db.total_changes - before)

    def samples(self, *, after_id: int = 0) -> list[StoredSample]:
        rows = self.db.execute(
            "SELECT id,feature_ts_ns,x_json,y FROM samples WHERE id>? ORDER BY id", (int(after_id),)
        ).fetchall()
        return [StoredSample(int(i), int(ts), Sample(tuple(json.loads(x)), int(y))) for i, ts, x, y in rows]

    def sample_count(self) -> int:
        return int(self.db.execute("SELECT count(*) FROM samples").fetchone()[0])

    def reset_learning_state(self) -> None:
        """Delete only derived learning state while preserving raw ticks and broker evidence.

        Use this after a feature/label definition change. Execution journals, positions and
        trade outcomes are intentionally untouched. The reset is explicit because it makes
        prior validation/model state incomparable with newly rebuilt samples.
        """
        learning_event_kinds = (
            "learning_validation_consumed",
            "learning_cycle",
            "model_promoted",
            "forward_qualification_updated",
        )
        with self.db:
            self.db.execute("DELETE FROM samples")
            self.db.execute("DELETE FROM sqlite_sequence WHERE name='samples'")
            exists = self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='sample_label_intervals'"
            ).fetchone()
            if exists:
                self.db.execute("DELETE FROM sample_label_intervals")
            self.db.execute(
                "DELETE FROM meta WHERE key IN (?, ?)",
                ("scalper_last_validation_sample_id", "scalper_champion_snapshot"),
            )
            placeholders = ",".join("?" for _ in learning_event_kinds)
            self.db.execute(
                f"DELETE FROM events WHERE kind IN ({placeholders})",
                learning_event_kinds,
            )
            self.db.execute(
                "DELETE FROM meta WHERE key LIKE 'scalper_forward_qualification_generation_%'"
            )

    def replace_positions(self, positions: Iterable[PositionState]) -> None:
        rows = list(positions)
        with self.db:
            self.db.execute("DELETE FROM positions")
            self.db.executemany(
                """INSERT INTO positions(
                       position_id,side,size,entry,opened_ns,peak_exit_price,trough_exit_price,reductions,broker_stop
                   ) VALUES(?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        p.position_id, p.side.value, p.size, p.entry, p.opened_ns,
                        p.peak_exit_price, p.trough_exit_price, p.reductions, p.broker_stop,
                    )
                    for p in rows
                ],
            )

    def load_positions(self) -> list[PositionState]:
        rows = self.db.execute(
            """SELECT position_id,side,size,entry,opened_ns,peak_exit_price,trough_exit_price,reductions,broker_stop
               FROM positions ORDER BY opened_ns,position_id"""
        ).fetchall()
        return [
            PositionState(str(pid), Side(side), float(size), float(entry), int(opened), peak, trough,
                          int(reductions), None if broker_stop is None else float(broker_stop))
            for pid, side, size, entry, opened, peak, trough, reductions, broker_stop in rows
        ]

    def append_event(self, ts_ns: int, kind: str, payload: dict) -> None:
        if not kind:
            raise ValueError("event kind required")
        document = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self.db:
            self.db.execute("INSERT INTO events(ts_ns,kind,payload_json) VALUES(?,?,?)",
                            (int(ts_ns), kind, document))

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
                                fraction: float, detail: str, stop: float | None = None,
                                model_generation: int = 0,
                                entry_equity: float | None = None) -> bool:
        if not decision_id:
            raise ValueError("decision_id required")
        with self.db:
            cur = self.db.execute(
                """INSERT OR IGNORE INTO execution_intents
                   (decision_id,created_ns,kind,position_id,side,size,fraction,stop,state,broker_position_id,
                    broker_identifier,model_generation,entry_equity,risk_amount,detail)
                   VALUES(?,?,?,?,?,?,?,?,'created',NULL,NULL,?,?,NULL,?)""",
                (decision_id, int(created_ns), kind, position_id, side, size, float(fraction), stop,
                 int(model_generation), entry_equity, detail),
            )
        return cur.rowcount == 1

    def transition_execution_intent(self, decision_id: str, state: str, *,
                                    broker_position_id: str | None = None,
                                    broker_identifier: int | None = None,
                                    risk_amount: float | None = None,
                                    detail: str | None = None) -> None:
        if state not in {"created", "submitted", "confirmed", "rejected", "unknown"}:
            raise ValueError("invalid execution state")
        current = self.execution_intent(decision_id)
        if current is None:
            raise KeyError(decision_id)
        allowed = {
            "created": {"submitted", "rejected"},
            "submitted": {"confirmed", "rejected", "unknown"},
            "unknown": {"confirmed", "rejected"},
            "confirmed": set(),
            "rejected": set(),
        }
        if state != current["state"] and state not in allowed[current["state"]]:
            raise ValueError("invalid execution transition")
        with self.db:
            self.db.execute(
                """UPDATE execution_intents
                   SET state=?,broker_position_id=COALESCE(?,broker_position_id),
                       broker_identifier=COALESCE(?,broker_identifier),
                       risk_amount=COALESCE(?,risk_amount),detail=COALESCE(?,detail)
                   WHERE decision_id=?""",
                (state, broker_position_id, broker_identifier, risk_amount, detail, decision_id),
            )

    def execution_intent(self, decision_id: str) -> dict | None:
        row = self.db.execute(
            """SELECT decision_id,created_ns,kind,position_id,side,size,fraction,stop,state,
                      broker_position_id,broker_identifier,model_generation,entry_equity,risk_amount,detail
               FROM execution_intents WHERE decision_id=?""",
            (decision_id,),
        ).fetchone()
        if row is None:
            return None
        keys = ("decision_id", "created_ns", "kind", "position_id", "side", "size", "fraction", "stop",
                "state", "broker_position_id", "broker_identifier", "model_generation", "entry_equity",
                "risk_amount", "detail")
        return dict(zip(keys, row))

    def unsettled_execution_intents(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT decision_id FROM execution_intents WHERE state IN ('created','submitted','unknown') ORDER BY created_ns"
        ).fetchall()
        return [self.execution_intent(str(row[0])) for row in rows]

    def confirmed_entry_intents_without_outcome(self) -> list[dict]:
        rows = self.db.execute(
            """SELECT decision_id FROM execution_intents e
               WHERE e.state='confirmed' AND e.kind IN ('OPEN','ADD')
                 AND NOT EXISTS(SELECT 1 FROM trade_outcomes t WHERE t.decision_id=e.decision_id)
               ORDER BY e.created_ns"""
        ).fetchall()
        return [self.execution_intent(str(row[0])) for row in rows]

    def record_trade_outcome(self, outcome: dict) -> bool:
        fields = (
            "position_id", "decision_id", "broker_identifier", "model_generation", "opened_ns", "closed_ns",
            "net_pnl", "profit", "commission", "swap", "fee", "volume", "risk_amount", "entry_equity",
        )
        values = tuple(outcome[k] for k in fields)
        with self.db:
            cur = self.db.execute(
                """INSERT OR IGNORE INTO trade_outcomes(
                       position_id,decision_id,broker_identifier,model_generation,opened_ns,closed_ns,
                       net_pnl,profit,commission,swap,fee,volume,risk_amount,entry_equity
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                values,
            )
        return cur.rowcount == 1

    def trade_outcomes(self, *, model_generation: int | None = None) -> list[dict]:
        sql = """SELECT position_id,decision_id,broker_identifier,model_generation,opened_ns,closed_ns,
                        net_pnl,profit,commission,swap,fee,volume,risk_amount,entry_equity
                 FROM trade_outcomes"""
        args: tuple[object, ...] = ()
        if model_generation is not None:
            sql += " WHERE model_generation=?"
            args = (int(model_generation),)
        sql += " ORDER BY closed_ns,position_id"
        keys = ("position_id", "decision_id", "broker_identifier", "model_generation", "opened_ns", "closed_ns",
                "net_pnl", "profit", "commission", "swap", "fee", "volume", "risk_amount", "entry_equity")
        return [dict(zip(keys, row)) for row in self.db.execute(sql, args).fetchall()]

    def meta(self, key: str, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if row is None else json.loads(row[0])

    def set_meta(self, key: str, value) -> None:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self.db:
            self.db.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, payload),
            )
