from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Iterable

import numpy as np

from .counterfactual import CounterfactualSettings
from .experience import EpisodicMemory, MarketEpisode
from .features import TickFeatureEngine
from .state_encoder import MarketState, MarketStateEncoder
from .store import ScalperStore
from .types import Tick


BUILDER_VERSION = "bbyg-v4-offline-episodes-1"
DEFAULT_MAX_GAP_NS = 300_000_000_000


@dataclass(frozen=True)
class OfflineEpisodeSettings:
    stride: int = 16
    horizon_ticks: int = 600
    risk_spreads: float = 2.0
    extra_cost_spreads: float = 0.20
    max_gap_ns: int = DEFAULT_MAX_GAP_NS
    outcome_chunk_size: int = 4096

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError("stride must be positive")
        CounterfactualSettings(
            horizon_ticks=self.horizon_ticks,
            risk_spreads=self.risk_spreads,
            extra_cost_spreads=self.extra_cost_spreads,
        )
        if self.max_gap_ns <= 0:
            raise ValueError("max_gap_ns must be positive")
        if self.outcome_chunk_size < 64:
            raise ValueError("outcome_chunk_size must be >=64")


@dataclass(frozen=True)
class OfflineEpisodeBuildReport:
    ticks: int
    segments: int
    episodes_generated: int
    episodes_persisted: int
    feature_dimensions: int
    digest: str
    dataset_signature: str
    regimes: dict[str, int]
    max_gap_ns: int
    stride: int
    horizon_ticks: int


@dataclass(frozen=True)
class _ResolvedStats:
    anchor_index: int
    resolved_ts_ns: int
    long_net_r: float
    long_mfe_r: float
    long_mae_r: float
    long_reward: float
    short_net_r: float
    short_mfe_r: float
    short_mae_r: float
    short_reward: float
    execution_cost_r: float


class OfflineEpisodeBuilder:
    """Build policy-independent BBYG v4 episodes from stored ticks.

    State encoding is causal.  Counterfactual outcomes are attached only after the full
    future horizon exists inside the same market segment.  The builder emits a deterministic
    binary digest so an independent second replay can verify exact state/outcome parity.
    """

    def __init__(self, settings: OfflineEpisodeSettings | None = None):
        self.settings = settings or OfflineEpisodeSettings()

    def dataset_signature(self) -> str:
        probe = MarketStateEncoder()
        document = {
            "builder_version": BUILDER_VERSION,
            "feature_names": probe.feature_names,
            "stride": self.settings.stride,
            "horizon_ticks": self.settings.horizon_ticks,
            "risk_spreads": self.settings.risk_spreads,
            "extra_cost_spreads": self.settings.extra_cost_spreads,
            "max_gap_ns": self.settings.max_gap_ns,
        }
        raw = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _segment_bounds(ticks: list[Tick], max_gap_ns: int) -> list[tuple[int, int]]:
        if not ticks:
            return []
        bounds: list[tuple[int, int]] = []
        begin = 0
        for i in range(1, len(ticks)):
            if ticks[i].ts_ns - ticks[i - 1].ts_ns > max_gap_ns:
                bounds.append((begin, i))
                begin = i
        bounds.append((begin, len(ticks)))
        return bounds

    @staticmethod
    def _reward_array(net_r: np.ndarray, mfe_r: np.ndarray, mae_r: np.ndarray,
                      cost_r: float) -> np.ndarray:
        regret_r = np.maximum(0.0, mfe_r - np.maximum(net_r, 0.0))
        z = (
            0.45 * net_r
            + 0.15 * mfe_r
            - 0.20 * np.abs(mae_r)
            - 0.10 * abs(cost_r)
            - 0.10 * np.abs(regret_r)
        )
        return np.tanh(z)

    def _resolve_segment(
        self,
        ticks: list[Tick],
        anchor_indices: list[int],
    ) -> dict[int, _ResolvedStats]:
        if not anchor_indices:
            return {}
        s = self.settings
        n = len(ticks)
        if any(i < 0 or i + s.horizon_ticks >= n for i in anchor_indices):
            raise ValueError("counterfactual anchor crosses segment boundary")

        bid = np.asarray([t.bid for t in ticks], dtype=np.float64)
        ask = np.asarray([t.ask for t in ticks], dtype=np.float64)
        ts = np.asarray([t.ts_ns for t in ticks], dtype=np.int64)
        anchors = np.asarray(anchor_indices, dtype=np.int64)
        path_len = s.horizon_ticks - 1
        if path_len < 1:
            raise ValueError("counterfactual path is empty")
        bid_windows = np.lib.stride_tricks.sliding_window_view(bid, path_len)
        ask_windows = np.lib.stride_tricks.sliding_window_view(ask, path_len)
        cost_r = float(s.extra_cost_spreads / s.risk_spreads)
        result: dict[int, _ResolvedStats] = {}

        for begin in range(0, len(anchors), s.outcome_chunk_size):
            anchor_chunk = anchors[begin:begin + s.outcome_chunk_size]
            entry_idx = anchor_chunk + 1
            path_start = anchor_chunk + 2
            resolved_idx = anchor_chunk + s.horizon_ticks

            entry_ask = ask[entry_idx]
            entry_bid = bid[entry_idx]
            entry_spread = np.maximum(entry_ask - entry_bid, 1e-12)
            risk_price = s.risk_spreads * entry_spread

            bid_path = bid_windows[path_start]
            ask_path = ask_windows[path_start]
            terminal_bid = bid[resolved_idx]
            terminal_ask = ask[resolved_idx]

            long_net = (terminal_bid - entry_ask) / risk_price - cost_r
            long_mfe = np.maximum(0.0, np.max(bid_path, axis=1) - entry_ask) / risk_price
            long_mae = np.maximum(0.0, entry_ask - np.min(bid_path, axis=1)) / risk_price
            long_reward = self._reward_array(long_net, long_mfe, long_mae, cost_r)

            short_net = (entry_bid - terminal_ask) / risk_price - cost_r
            short_mfe = np.maximum(0.0, entry_bid - np.min(ask_path, axis=1)) / risk_price
            short_mae = np.maximum(0.0, np.max(ask_path, axis=1) - entry_bid) / risk_price
            short_reward = self._reward_array(short_net, short_mfe, short_mae, cost_r)

            for j, anchor_index in enumerate(anchor_chunk.tolist()):
                result[int(anchor_index)] = _ResolvedStats(
                    anchor_index=int(anchor_index),
                    resolved_ts_ns=int(ts[resolved_idx[j]]),
                    long_net_r=float(long_net[j]),
                    long_mfe_r=float(long_mfe[j]),
                    long_mae_r=float(long_mae[j]),
                    long_reward=float(long_reward[j]),
                    short_net_r=float(short_net[j]),
                    short_mfe_r=float(short_mfe[j]),
                    short_mae_r=float(short_mae[j]),
                    short_reward=float(short_reward[j]),
                    execution_cost_r=cost_r,
                )
        return result

    @staticmethod
    def _digest_episode(hasher, episode: MarketEpisode) -> None:
        hasher.update(struct.pack(">q", int(episode.ts_ns)))
        hasher.update(episode.regime.encode("utf-8") + b"\0")
        hasher.update(episode.proposal.encode("utf-8") + b"\0")
        hasher.update(struct.pack(">?", bool(episode.executed)))
        for value in episode.state_embedding:
            hasher.update(struct.pack(">d", float(value)))
        for value in (
            episode.realized_reward,
            episode.counterfactual_long_reward,
            episode.counterfactual_short_reward,
            episode.mfe_r,
            episode.mae_r,
            episode.execution_cost_r,
        ):
            if value is None:
                hasher.update(b"N")
            else:
                hasher.update(b"F" + struct.pack(">d", float(value)))
        context = json.dumps(
            episode.context, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        hasher.update(struct.pack(">I", len(context)))
        hasher.update(context)

    @staticmethod
    def _persist_batch(store: ScalperStore, episodes: list[MarketEpisode]) -> int:
        if not episodes:
            return 0
        rows = [
            (
                int(e.ts_ns),
                json.dumps(list(e.state_embedding), separators=(",", ":"), allow_nan=False),
                json.dumps(list(e.connectome_embedding), separators=(",", ":"), allow_nan=False),
                e.regime,
                e.proposal,
                int(e.executed),
                e.realized_reward,
                e.counterfactual_long_reward,
                e.counterfactual_short_reward,
                e.mfe_r,
                e.mae_r,
                e.execution_cost_r,
                json.dumps(e.context, sort_keys=True, separators=(",", ":"), allow_nan=False),
            )
            for e in episodes
        ]
        before = store.db.total_changes
        with store.db:
            store.db.executemany(
                """INSERT OR IGNORE INTO market_episodes(
                       ts_ns,state_embedding_json,connectome_embedding_json,regime,proposal,executed,
                       realized_reward,counterfactual_long_reward,counterfactual_short_reward,
                       mfe_r,mae_r,execution_cost_r,context_json
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        return int(store.db.total_changes - before)

    def _segment_episodes(self, ticks: list[Tick]) -> list[MarketEpisode]:
        s = self.settings
        if len(ticks) <= s.horizon_ticks + 1:
            return []
        micro = TickFeatureEngine(window=96, min_ticks=96, fast_ticks=8, slow_ticks=24)
        encoder = MarketStateEncoder()
        states: dict[int, MarketState] = {}

        for i, tick in enumerate(ticks):
            mf = micro.update(tick)
            eligible = (
                i % s.stride == 0
                and i + s.horizon_ticks < len(ticks)
            )
            state = encoder.update(tick, mf, compute=eligible)
            if state is not None:
                states[i] = state

        resolved = self._resolve_segment(ticks, list(states))
        episodes: list[MarketEpisode] = []
        for i in sorted(states):
            state = states[i]
            outcome = resolved[i]
            context = dict(state.context)
            context.update({
                "builder_version": BUILDER_VERSION,
                "resolved_ts_ns": outcome.resolved_ts_ns,
                "ticks_observed": s.horizon_ticks,
                "long_net_r": outcome.long_net_r,
                "long_mfe_r": outcome.long_mfe_r,
                "long_mae_r": outcome.long_mae_r,
                "short_net_r": outcome.short_net_r,
                "short_mfe_r": outcome.short_mfe_r,
                "short_mae_r": outcome.short_mae_r,
                "offline_policy_independent": True,
            })
            episodes.append(MarketEpisode(
                ts_ns=state.ts_ns,
                state_embedding=state.embedding,
                regime=state.regime,
                proposal="FLAT",
                executed=False,
                counterfactual_long_reward=outcome.long_reward,
                counterfactual_short_reward=outcome.short_reward,
                mfe_r=max(outcome.long_mfe_r, outcome.short_mfe_r),
                mae_r=max(outcome.long_mae_r, outcome.short_mae_r),
                execution_cost_r=outcome.execution_cost_r,
                context=context,
            ))
        return episodes

    def build(
        self,
        store: ScalperStore,
        *,
        persist: bool,
        replace: bool = False,
        ticks: list[Tick] | None = None,
    ) -> OfflineEpisodeBuildReport:
        memory = EpisodicMemory(store)
        signature = self.dataset_signature()
        existing_signature = store.meta("v4_episode_dataset_signature", None)
        existing_count = memory.count()

        if persist and existing_count and existing_signature != signature and not replace:
            raise ValueError(
                "existing v4 episodes were built with another dataset signature; use replace=True"
            )
        if persist and replace:
            with store.db:
                store.db.execute("DELETE FROM market_episodes")
            existing_count = 0

        rows = store.ticks() if ticks is None else list(ticks)
        bounds = self._segment_bounds(rows, self.settings.max_gap_ns)
        digest = hashlib.sha256()
        digest.update(signature.encode("ascii"))
        generated = 0
        persisted = 0
        regimes: dict[str, int] = {}

        for begin, end in bounds:
            segment = rows[begin:end]
            episodes = self._segment_episodes(segment)
            for episode in episodes:
                self._digest_episode(digest, episode)
                regimes[episode.regime] = regimes.get(episode.regime, 0) + 1
            generated += len(episodes)
            if persist:
                persisted += self._persist_batch(store, episodes)

        report = OfflineEpisodeBuildReport(
            ticks=len(rows),
            segments=len(bounds),
            episodes_generated=generated,
            episodes_persisted=persisted,
            feature_dimensions=MarketStateEncoder().dimensions,
            digest=digest.hexdigest(),
            dataset_signature=signature,
            regimes=dict(sorted(regimes.items())),
            max_gap_ns=self.settings.max_gap_ns,
            stride=self.settings.stride,
            horizon_ticks=self.settings.horizon_ticks,
        )
        if persist:
            store.set_meta("v4_episode_dataset_signature", signature)
            store.set_meta("v4_episode_dataset_digest", report.digest)
            store.set_meta("v4_episode_dataset_count", generated)
            store.set_meta("v4_episode_builder_version", BUILDER_VERSION)
        return report

    def parity_audit(
        self,
        store: ScalperStore,
        *,
        ticks: list[Tick] | None = None,
    ) -> dict:
        expected_digest = store.meta("v4_episode_dataset_digest", None)
        expected_signature = store.meta("v4_episode_dataset_signature", None)
        expected_count = store.meta("v4_episode_dataset_count", None)
        if expected_digest is None or expected_signature is None or expected_count is None:
            raise ValueError("no persisted v4 episode dataset metadata to audit")
        report = self.build(store, persist=False, ticks=ticks)
        return {
            "pass": (
                report.dataset_signature == expected_signature
                and report.digest == expected_digest
                and report.episodes_generated == int(expected_count)
            ),
            "expected_signature": expected_signature,
            "actual_signature": report.dataset_signature,
            "expected_digest": expected_digest,
            "actual_digest": report.digest,
            "expected_count": int(expected_count),
            "actual_count": report.episodes_generated,
            "segments": report.segments,
            "ticks": report.ticks,
            "regimes": report.regimes,
        }
