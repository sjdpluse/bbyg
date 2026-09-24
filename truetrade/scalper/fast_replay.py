from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .labels import CostAwareLabeler
from .learning import Sample
from .sample_intervals import SampleLabelInterval, record_label_intervals
from .store import ScalperStore


@dataclass(frozen=True)
class FastReplayReport:
    ticks: int
    feature_rows: int
    labeled: int
    long_labels: int
    short_labels: int
    skipped: int
    market_gaps: int
    max_gap_seconds: float
    stride: int
    extra_cost_spreads: float
    stop_reference: str
    max_entry_delay_seconds: float
    nominal_target_from_entry_spreads: float
    nominal_stop_from_entry_spreads: float


class FastGapAwareReplayBuilder:
    """Vectorized replay for multi-million-tick research datasets.

    Features are observed at a causal decision tick. A hypothetical order may enter only
    on the first strictly later tick, matching economic/demo execution. Target/stop paths
    are then evaluated from that executable entry quote. A full 96-tick feature warmup is
    required after startup or a market gap, and label evidence never crosses a market gap.
    """

    FEATURE_WINDOW = 96
    FAST_TICKS = 8
    SLOW_TICKS = 24
    MOVE_CLIP = 8.0
    VOLATILITY_CLIP = 4.0
    SPREAD_Z_CLIP = 8.0

    def __init__(
        self,
        labeler: CostAwareLabeler | None = None,
        *,
        stride: int = 4,
        max_gap_seconds: float = 300.0,
        feature_batch: int = 20_000,
        label_batch: int = 2_500,
        write_batch: int = 10_000,
    ):
        if stride < 1:
            raise ValueError("stride must be positive")
        if max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be positive")
        if min(feature_batch, label_batch, write_batch) < 1:
            raise ValueError("batch sizes must be positive")
        self.labeler = labeler or CostAwareLabeler()
        self.stride = int(stride)
        self.max_gap_seconds = float(max_gap_seconds)
        self.feature_batch = int(feature_batch)
        self.label_batch = int(label_batch)
        self.write_batch = int(write_batch)

    @staticmethod
    def _load_arrays(store: ScalperStore) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = int(store.db.execute("SELECT count(*) FROM ticks").fetchone()[0])
        if count == 0:
            return (np.empty(0, dtype=np.int64), np.empty(0), np.empty(0))
        dtype = np.dtype([("ts", "<i8"), ("bid", "<f8"), ("ask", "<f8")])
        rows = np.fromiter(
            store.db.execute("SELECT ts_ns,bid,ask FROM ticks ORDER BY ts_ns"),
            dtype=dtype,
            count=count,
        )
        return rows["ts"], rows["bid"], rows["ask"]

    def _anchors(self, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(ts)
        if n < self.FEATURE_WINDOW:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        gap_ns = int(self.max_gap_seconds * 1_000_000_000)
        gap_starts = np.flatnonzero(np.diff(ts) > gap_ns).astype(np.int64) + 1
        markers = np.zeros(n, dtype=np.int64)
        markers[gap_starts] = gap_starts
        segment_start = np.maximum.accumulate(markers)
        anchors = np.arange(0, n, self.stride, dtype=np.int64)
        warm = anchors - segment_start[anchors] >= self.FEATURE_WINDOW - 1
        return anchors[warm], gap_starts

    @classmethod
    def _feature_matrix(
        cls,
        anchors: np.ndarray,
        bid: np.ndarray,
        ask: np.ndarray,
    ) -> np.ndarray:
        if len(anchors) == 0:
            return np.empty((0, 8), dtype=float)
        mids = (bid + ask) * 0.5
        spreads = np.maximum(ask - bid, 1e-12)
        diffs = np.diff(mids)
        move_units = np.clip(diffs / spreads[1:], -cls.MOVE_CLIP, cls.MOVE_CLIP)

        slow_offsets = np.arange(-cls.SLOW_TICKS, 0, dtype=np.int64)
        fast_offsets = np.arange(-cls.FAST_TICKS, 0, dtype=np.int64)
        hist_offsets = np.arange(-(cls.FEATURE_WINDOW - 1), 0, dtype=np.int64)

        slow_moves = move_units[anchors[:, None] + slow_offsets]
        fast_moves = move_units[anchors[:, None] + fast_offsets]
        recent = diffs[anchors[:, None] + slow_offsets]
        hist_spreads = spreads[anchors[:, None] + hist_offsets]
        current_spread = spreads[anchors]

        fast = np.mean(fast_moves, axis=1)
        slow = np.mean(slow_moves, axis=1)
        acceleration = np.clip(fast - slow, -cls.MOVE_CLIP, cls.MOVE_CLIP)
        imbalance = np.mean(np.sign(recent), axis=1)
        gross = np.sum(np.abs(recent), axis=1)
        net = np.sum(recent, axis=1)
        efficiency = np.divide(net, gross, out=np.zeros_like(net), where=gross > 1e-12)
        efficiency = np.clip(efficiency, -1.0, 1.0)
        noise = np.sqrt(np.mean(recent * recent, axis=1))
        volatility_ratio = np.clip(noise / current_spread, 0.0, cls.VOLATILITY_CLIP)
        last_move_ratio = np.clip(diffs[anchors - 1] / current_spread, -cls.MOVE_CLIP, cls.MOVE_CLIP)

        median = np.median(hist_spreads, axis=1)
        mad = np.median(np.abs(hist_spreads - median[:, None]), axis=1)
        robust_scale = np.maximum.reduce((1.4826 * mad, median * 0.02, np.full_like(median, 1e-12)))
        spread_z = np.clip((current_spread - median) / robust_scale,
                           -cls.SPREAD_Z_CLIP, cls.SPREAD_Z_CLIP)

        matrix = np.column_stack((
            spread_z,
            fast,
            slow,
            acceleration,
            imbalance,
            efficiency,
            volatility_ratio,
            last_move_ratio,
        ))
        if not np.isfinite(matrix).all():
            raise ValueError("non-finite vectorized feature")
        return matrix

    @staticmethod
    def _first_true(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
        event = mask & valid
        has = np.any(event, axis=1)
        first = np.argmax(event, axis=1).astype(np.int64)
        first[~has] = np.iinfo(np.int64).max
        return first

    def _label_batch(
        self,
        anchors: np.ndarray,
        vectors: np.ndarray,
        ts: np.ndarray,
        bid: np.ndarray,
        ask: np.ndarray,
        gap_starts: np.ndarray,
    ) -> tuple[list[tuple[int, Sample]], list[SampleLabelInterval], int, int, int]:
        if len(anchors) == 0:
            return [], [], 0, 0, 0
        s = self.labeler.settings
        n = len(ts)
        max_future = int(s.max_lookahead_ticks)
        future_offsets = np.arange(1, max_future + 1, dtype=np.int64)

        # Decision at anchor -> first strictly later tick is the executable entry.
        entry = anchors + 1
        entry_exists = entry < n
        entry_delay_ok = np.zeros(len(anchors), dtype=bool)
        if np.any(entry_exists):
            delay_ns = ts[entry[entry_exists]] - ts[anchors[entry_exists]]
            entry_delay_ok[entry_exists] = delay_ns <= int(s.max_entry_delay_seconds * 1_000_000_000)

        # An entry may not jump across a market closure/gap.
        next_gap_pos = np.searchsorted(gap_starts, anchors, side="right")
        next_gap = np.full(len(anchors), n, dtype=np.int64)
        has_gap = next_gap_pos < len(gap_starts)
        if np.any(has_gap):
            next_gap[has_gap] = gap_starts[next_gap_pos[has_gap]]
        entry_before_gap = entry < next_gap
        executable = entry_exists & entry_delay_ok & entry_before_gap

        # Label evidence starts after entry and is capped at the next market gap.
        future_count = np.zeros(len(anchors), dtype=np.int64)
        if np.any(executable):
            future_count[executable] = np.minimum.reduce((
                np.full(int(executable.sum()), max_future, dtype=np.int64),
                next_gap[executable] - entry[executable] - 1,
                np.full(int(executable.sum()), n, dtype=np.int64) - entry[executable] - 1,
            ))
        enough = executable & (future_count >= 10)
        skipped = int((~enough).sum())
        if not np.any(enough):
            return [], [], skipped, 0, 0

        decision = anchors[enough]
        e = entry[enough]
        v = vectors[enough]
        counts = future_count[enough]
        idx = e[:, None] + future_offsets[None, :]
        safe_idx = np.minimum(idx, n - 1)
        valid = future_offsets[None, :] <= counts[:, None]
        fb = bid[safe_idx]
        fa = ask[safe_idx]

        spread = np.maximum(ask[e] - bid[e], 1e-12)
        long_profit = ask[e] + (s.profit_spreads + s.extra_cost_spreads) * spread
        long_stop = s.long_stop_price(bid[e], ask[e], spread)
        short_profit = bid[e] - (s.profit_spreads + s.extra_cost_spreads) * spread
        short_stop = s.short_stop_price(bid[e], ask[e], spread)

        first_long_target = self._first_true(fb >= long_profit[:, None], valid)
        first_long_stop = self._first_true(fb <= long_stop[:, None], valid)
        first_short_target = self._first_true(fa <= short_profit[:, None], valid)
        first_short_stop = self._first_true(fa >= short_stop[:, None], valid)

        long_wins = first_long_target < first_long_stop
        short_wins = first_short_target < first_short_stop
        both_win = long_wins & short_wins
        long_first = long_wins & (~short_wins | (first_long_target < first_short_target))
        short_first = short_wins & (~long_wins | (first_short_target < first_long_target))
        ambiguous = both_win & (first_long_target == first_short_target)
        labeled = (long_first | short_first) & ~ambiguous
        skipped += int((~labeled).sum())

        winning_index = np.where(long_first, first_long_target, first_short_target)
        sample_rows: list[tuple[int, Sample]] = []
        interval_rows: list[SampleLabelInterval] = []
        for j in np.flatnonzero(labeled):
            label = 1 if bool(long_first[j]) else 0
            decision_index = int(decision[j])
            entry_index = int(e[j])
            event_zero_based = int(winning_index[j])
            end_index = entry_index + event_zero_based + 1
            ticks_observed = end_index - decision_index
            feature_ts = int(ts[decision_index])
            end_ts = int(ts[end_index])
            sample_rows.append((feature_ts, Sample(tuple(float(x) for x in v[j]), label)))
            interval_rows.append(SampleLabelInterval(feature_ts, end_ts, ticks_observed))
        return sample_rows, interval_rows, skipped, int((long_first & ~ambiguous).sum()), int((short_first & ~ambiguous).sum())

    def build(self, store: ScalperStore, *, reset_learning: bool = True, progress=None) -> FastReplayReport:
        if reset_learning:
            store.reset_learning_state()
        ts, bid, ask = self._load_arrays(store)
        n = len(ts)
        if n < self.FEATURE_WINDOW + 11:
            raise ValueError("not enough ticks for replay")
        anchors, gap_starts = self._anchors(ts)
        if progress is not None:
            progress("loaded", ticks=n, feature_candidates=len(anchors), market_gaps=len(gap_starts))

        labeled = 0
        long_labels = 0
        short_labels = 0
        skipped = 0
        processed = 0
        for start in range(0, len(anchors), self.feature_batch):
            batch_anchors = anchors[start:start + self.feature_batch]
            vectors = self._feature_matrix(batch_anchors, bid, ask)
            batch_samples: list[tuple[int, Sample]] = []
            batch_intervals: list[SampleLabelInterval] = []
            for sub in range(0, len(batch_anchors), self.label_batch):
                end = sub + self.label_batch
                samples, intervals, sub_skipped, sub_long, sub_short = self._label_batch(
                    batch_anchors[sub:end], vectors[sub:end], ts, bid, ask, gap_starts
                )
                batch_samples.extend(samples)
                batch_intervals.extend(intervals)
                skipped += sub_skipped
                long_labels += sub_long
                short_labels += sub_short

            for write_start in range(0, len(batch_samples), self.write_batch):
                write_end = write_start + self.write_batch
                labeled += store.add_samples(batch_samples[write_start:write_end])
                record_label_intervals(store, batch_intervals[write_start:write_end])

            processed += len(batch_anchors)
            if progress is not None:
                progress(
                    "replay_progress",
                    processed_feature_candidates=processed,
                    total_feature_candidates=len(anchors),
                    samples=store.sample_count(),
                    skipped=skipped,
                )

        if labeled != store.sample_count():
            labeled = store.sample_count()
        return FastReplayReport(
            ticks=n,
            feature_rows=len(anchors),
            labeled=labeled,
            long_labels=long_labels,
            short_labels=short_labels,
            skipped=skipped,
            market_gaps=len(gap_starts),
            max_gap_seconds=self.max_gap_seconds,
            stride=self.stride,
            extra_cost_spreads=self.labeler.settings.extra_cost_spreads,
            stop_reference=self.labeler.settings.stop_reference,
            max_entry_delay_seconds=self.labeler.settings.max_entry_delay_seconds,
            nominal_target_from_entry_spreads=self.labeler.settings.nominal_target_from_entry_spreads,
            nominal_stop_from_entry_spreads=self.labeler.settings.nominal_stop_from_entry_spreads,
        )
