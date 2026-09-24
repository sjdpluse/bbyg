from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Iterable

import numpy as np

from .store import ScalperStore
from .types import Side


@dataclass(frozen=True)
class RewardComponents:
    """Normalized outcome components used for reward-modulated memory.

    Values are expressed in R units where possible.  The coefficients are intentionally
    explicit research parameters rather than hidden constants inside the memory store.
    """

    net_r: float
    mfe_r: float = 0.0
    mae_r: float = 0.0
    execution_cost_r: float = 0.0
    regret_r: float = 0.0

    def __post_init__(self) -> None:
        values = (self.net_r, self.mfe_r, self.mae_r, self.execution_cost_r, self.regret_r)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("reward components must be finite")


def bounded_reward(
    components: RewardComponents,
    *,
    net_weight: float = 0.45,
    mfe_weight: float = 0.15,
    mae_weight: float = 0.20,
    cost_weight: float = 0.10,
    regret_weight: float = 0.10,
) -> float:
    """Return a bounded artificial-dopamine signal in [-1, 1]."""

    weights = (net_weight, mfe_weight, mae_weight, cost_weight, regret_weight)
    if not all(math.isfinite(v) and v >= 0 for v in weights):
        raise ValueError("reward weights must be finite and non-negative")
    z = (
        net_weight * components.net_r
        + mfe_weight * components.mfe_r
        - mae_weight * abs(components.mae_r)
        - cost_weight * abs(components.execution_cost_r)
        - regret_weight * abs(components.regret_r)
    )
    return float(math.tanh(z))


@dataclass(frozen=True)
class MarketEpisode:
    ts_ns: int
    state_embedding: tuple[float, ...]
    regime: str
    proposal: str
    executed: bool = False
    connectome_embedding: tuple[float, ...] = ()
    realized_reward: float | None = None
    counterfactual_long_reward: float | None = None
    counterfactual_short_reward: float | None = None
    mfe_r: float | None = None
    mae_r: float | None = None
    execution_cost_r: float | None = None
    context: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ts_ns <= 0:
            raise ValueError("episode timestamp must be positive")
        if not self.state_embedding:
            raise ValueError("state embedding required")
        if not all(math.isfinite(float(v)) for v in self.state_embedding):
            raise ValueError("state embedding must be finite")
        if self.connectome_embedding and not all(
            math.isfinite(float(v)) for v in self.connectome_embedding
        ):
            raise ValueError("connectome embedding must be finite")
        if not self.regime:
            raise ValueError("regime required")
        if self.proposal not in {Side.LONG.value, Side.SHORT.value, "FLAT"}:
            raise ValueError("proposal must be LONG, SHORT or FLAT")
        for value in (
            self.realized_reward,
            self.counterfactual_long_reward,
            self.counterfactual_short_reward,
            self.mfe_r,
            self.mae_r,
            self.execution_cost_r,
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("episode outcomes must be finite")


@dataclass(frozen=True)
class EpisodeMatch:
    episode_id: int
    similarity: float
    episode: MarketEpisode


@dataclass(frozen=True)
class RecallSummary:
    matches: tuple[EpisodeMatch, ...]
    weighted_realized_reward: float | None
    weighted_long_reward: float | None
    weighted_short_reward: float | None
    positive_fraction: float | None
    memory_confidence: float

    @property
    def long_minus_short_bias(self) -> float | None:
        if self.weighted_long_reward is None or self.weighted_short_reward is None:
            return None
        return float(self.weighted_long_reward - self.weighted_short_reward)


class EpisodicMemory:
    """Durable similarity memory over market-state embeddings.

    The store deliberately keeps raw embeddings and performs exact cosine recall first.
    This is slower than an ANN index but deterministic, auditable and suitable for the
    initial BBYG v4 research phase.  An approximate index may be introduced later while
    keeping this table as the source of truth.
    """

    def __init__(self, store: ScalperStore):
        self.store = store
        self.store.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS market_episodes(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ns INTEGER NOT NULL UNIQUE,
                state_embedding_json TEXT NOT NULL,
                connectome_embedding_json TEXT NOT NULL,
                regime TEXT NOT NULL,
                proposal TEXT NOT NULL CHECK(proposal IN ('LONG','SHORT','FLAT')),
                executed INTEGER NOT NULL CHECK(executed IN (0,1)),
                realized_reward REAL,
                counterfactual_long_reward REAL,
                counterfactual_short_reward REAL,
                mfe_r REAL,
                mae_r REAL,
                execution_cost_r REAL,
                context_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_market_episodes_regime_ts
                ON market_episodes(regime, ts_ns);
            """
        )
        self.store.db.commit()

    @staticmethod
    def _json_vector(values: Iterable[float]) -> str:
        return json.dumps([float(v) for v in values], separators=(",", ":"), allow_nan=False)

    @staticmethod
    def _row_to_episode(row) -> tuple[int, MarketEpisode]:
        (
            episode_id, ts_ns, state_json, connectome_json, regime, proposal, executed,
            realized, cf_long, cf_short, mfe_r, mae_r, cost_r, context_json,
        ) = row
        episode = MarketEpisode(
            ts_ns=int(ts_ns),
            state_embedding=tuple(float(v) for v in json.loads(state_json)),
            connectome_embedding=tuple(float(v) for v in json.loads(connectome_json)),
            regime=str(regime),
            proposal=str(proposal),
            executed=bool(executed),
            realized_reward=None if realized is None else float(realized),
            counterfactual_long_reward=None if cf_long is None else float(cf_long),
            counterfactual_short_reward=None if cf_short is None else float(cf_short),
            mfe_r=None if mfe_r is None else float(mfe_r),
            mae_r=None if mae_r is None else float(mae_r),
            execution_cost_r=None if cost_r is None else float(cost_r),
            context=json.loads(context_json),
        )
        return int(episode_id), episode

    def add(self, episode: MarketEpisode) -> int:
        with self.store.db:
            cur = self.store.db.execute(
                """INSERT INTO market_episodes(
                       ts_ns,state_embedding_json,connectome_embedding_json,regime,proposal,executed,
                       realized_reward,counterfactual_long_reward,counterfactual_short_reward,
                       mfe_r,mae_r,execution_cost_r,context_json
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    int(episode.ts_ns),
                    self._json_vector(episode.state_embedding),
                    self._json_vector(episode.connectome_embedding),
                    episode.regime,
                    episode.proposal,
                    int(episode.executed),
                    episode.realized_reward,
                    episode.counterfactual_long_reward,
                    episode.counterfactual_short_reward,
                    episode.mfe_r,
                    episode.mae_r,
                    episode.execution_cost_r,
                    json.dumps(episode.context, sort_keys=True, separators=(",", ":"), allow_nan=False),
                ),
            )
        return int(cur.lastrowid)

    def update_outcome(
        self,
        episode_id: int,
        *,
        executed: bool | None = None,
        realized_reward: float | None = None,
        counterfactual_long_reward: float | None = None,
        counterfactual_short_reward: float | None = None,
        mfe_r: float | None = None,
        mae_r: float | None = None,
        execution_cost_r: float | None = None,
    ) -> None:
        values = (
            realized_reward,
            counterfactual_long_reward,
            counterfactual_short_reward,
            mfe_r,
            mae_r,
            execution_cost_r,
        )
        if any(v is not None and not math.isfinite(float(v)) for v in values):
            raise ValueError("outcome values must be finite")
        row = self.store.db.execute("SELECT id FROM market_episodes WHERE id=?", (int(episode_id),)).fetchone()
        if row is None:
            raise KeyError(episode_id)
        with self.store.db:
            self.store.db.execute(
                """UPDATE market_episodes SET
                       executed=COALESCE(?,executed),
                       realized_reward=COALESCE(?,realized_reward),
                       counterfactual_long_reward=COALESCE(?,counterfactual_long_reward),
                       counterfactual_short_reward=COALESCE(?,counterfactual_short_reward),
                       mfe_r=COALESCE(?,mfe_r),
                       mae_r=COALESCE(?,mae_r),
                       execution_cost_r=COALESCE(?,execution_cost_r)
                   WHERE id=?""",
                (
                    None if executed is None else int(executed),
                    realized_reward,
                    counterfactual_long_reward,
                    counterfactual_short_reward,
                    mfe_r,
                    mae_r,
                    execution_cost_r,
                    int(episode_id),
                ),
            )

    def get(self, episode_id: int) -> MarketEpisode | None:
        row = self.store.db.execute(
            """SELECT id,ts_ns,state_embedding_json,connectome_embedding_json,regime,proposal,executed,
                      realized_reward,counterfactual_long_reward,counterfactual_short_reward,
                      mfe_r,mae_r,execution_cost_r,context_json
               FROM market_episodes WHERE id=?""",
            (int(episode_id),),
        ).fetchone()
        return None if row is None else self._row_to_episode(row)[1]

    def count(self) -> int:
        return int(self.store.db.execute("SELECT count(*) FROM market_episodes").fetchone()[0])

    def recall(
        self,
        query_embedding: Iterable[float],
        *,
        k: int = 32,
        regime: str | None = None,
        before_ts_ns: int | None = None,
        min_similarity: float = -1.0,
    ) -> RecallSummary:
        if k < 1:
            raise ValueError("k must be positive")
        query = np.asarray(tuple(query_embedding), dtype=float)
        if query.ndim != 1 or len(query) == 0 or not np.isfinite(query).all():
            raise ValueError("finite query embedding required")
        norm = float(np.linalg.norm(query))
        if norm <= 1e-12:
            raise ValueError("query embedding must have non-zero norm")

        sql = (
            "SELECT id,ts_ns,state_embedding_json,connectome_embedding_json,regime,proposal,executed,"
            "realized_reward,counterfactual_long_reward,counterfactual_short_reward,"
            "mfe_r,mae_r,execution_cost_r,context_json FROM market_episodes WHERE 1=1"
        )
        args: list[object] = []
        if regime is not None:
            sql += " AND regime=?"
            args.append(regime)
        if before_ts_ns is not None:
            sql += " AND ts_ns<?"
            args.append(int(before_ts_ns))
        rows = self.store.db.execute(sql, args).fetchall()

        scored: list[EpisodeMatch] = []
        for row in rows:
            episode_id, episode = self._row_to_episode(row)
            vector = np.asarray(episode.state_embedding, dtype=float)
            if vector.shape != query.shape:
                continue
            denominator = norm * float(np.linalg.norm(vector))
            if denominator <= 1e-12:
                continue
            similarity = float(np.dot(query, vector) / denominator)
            if similarity >= min_similarity:
                scored.append(EpisodeMatch(episode_id, similarity, episode))
        scored.sort(key=lambda item: (-item.similarity, -item.episode.ts_ns, item.episode_id))
        matches = tuple(scored[:k])

        if not matches:
            return RecallSummary((), None, None, None, None, 0.0)
        raw_weights = np.asarray([max(0.0, m.similarity) ** 2 for m in matches], dtype=float)
        if float(raw_weights.sum()) <= 1e-12:
            raw_weights = np.ones(len(matches), dtype=float)
        weights = raw_weights / raw_weights.sum()

        def weighted(field_name: str) -> float | None:
            pairs = [
                (weights[i], getattr(match.episode, field_name))
                for i, match in enumerate(matches)
                if getattr(match.episode, field_name) is not None
            ]
            if not pairs:
                return None
            denom = sum(w for w, _ in pairs)
            return float(sum(w * float(v) for w, v in pairs) / denom)

        realized = weighted("realized_reward")
        long_reward = weighted("counterfactual_long_reward")
        short_reward = weighted("counterfactual_short_reward")
        resolved_realized = [m.episode.realized_reward for m in matches if m.episode.realized_reward is not None]
        positive_fraction = (
            None if not resolved_realized
            else float(sum(float(v) > 0 for v in resolved_realized) / len(resolved_realized))
        )
        similarity_strength = float(np.clip(np.mean([max(0.0, m.similarity) for m in matches]), 0.0, 1.0))
        resolved_fraction = float(
            sum(
                m.episode.realized_reward is not None
                or m.episode.counterfactual_long_reward is not None
                or m.episode.counterfactual_short_reward is not None
                for m in matches
            ) / len(matches)
        )
        coverage = min(1.0, len(matches) / max(8.0, float(k)))
        confidence = float(np.clip(similarity_strength * resolved_fraction * coverage, 0.0, 1.0))
        return RecallSummary(
            matches,
            realized,
            long_reward,
            short_reward,
            positive_fraction,
            confidence,
        )

    def snapshot(self) -> dict:
        row = self.store.db.execute(
            """SELECT count(*),
                      sum(CASE WHEN realized_reward IS NOT NULL THEN 1 ELSE 0 END),
                      avg(realized_reward)
               FROM market_episodes"""
        ).fetchone()
        return {
            "episodes": int(row[0] or 0),
            "resolved_realized": int(row[1] or 0),
            "mean_realized_reward": None if row[2] is None else float(row[2]),
        }
