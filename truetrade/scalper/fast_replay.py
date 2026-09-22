from __future__ import annotations

from dataclasses import dataclass
import math

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


class FastGapAwareReplayBuilder:
    """Vectorized replay for multi-million-tick research datasets.

    It keeps the live feature definitions but requires a full 96-tick feature warmup after
    startup or a market gap. Label evidence is capped at the first market gap, so a Friday
    close can never use Sunday/Monday quotes to resolve a seconds-scalping label.
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
        offsets = np.arange(1, max_future + 1, dtype=np.int64)

        next_gap_pos = np.searchsorted(gap_starts, anchors, side="right")
        next_gap = np.full(len(anchors), n, dtype=np.int64)
        has_gap = next_gap_pos < len(gap_starts)
        if np.any(has_gap):
            next_gap[has_gap] = gap_starts[next_gap_pos[has_gap]]
        future_count = np.minimum.reduce((
            np.full(len(anchors), max_future, dtype=np.int64),
            next_gap - anchors - 1,
            np.full(len(anchors), n, dtype=np.int64) - anchors - 1,
        ))
        enough = future_count >= 10
        skipped = int((~enough).sum())
        if not np.any(enough):
            return [], [], skipped, 0, 0

        a = anchors[enough]
        v = vectors[enough]
        counts = future_count[enough]
        idx = a[:, None] + offsets[None, :]
        safe_idx = np.minimum(idx, n - 1)
        valid = offsets[None, :] <= counts[:, None]
        fb = bid[safe_idx]
        fa = ask[safe_idx]

        spread = np.maximum(ask[a] - bid[a], 1e-12)
        long_profit = ask[a] + (s.profit_spreads + s.extra_cost_spreads) * spread
        long_loss = bid[a] - s.loss_spreads * spread
        short_profit = bid[a] - (s.profit_spreads + s.extra_cost_spreads) * spread
        short_loss = ask[a] + s.loss_spreads * spread

        long_win = (fb >= long_profit[:, None]) & valid
        short_win = (fa <= short_profit[:, None]) & valid
        adverse = (fb <= long_loss[:, None]) & (fa >= short_loss[:, None]) & valid
        event = long_win | short_win | adverse
        has_event = np.any(event, axis=1)
        first = np.argmax(event, axis=1)
        row_idx = np.arange(len(a))
        first_long = long_win[row_idx, first] & has_event
        first_short = short_win[row_idx, first] & has_event
        ambiguous_profit = first_long & first_short
        long_label = first_long & ~ambiguous_profit
        short_label = first_short & ~first_long
        labeled = long_label | short_label
        skipped += int((~labeled).sum())

        observed = first + 1
        sample_rows: list[tuple[int, Sample]] = []
        interval_rows: list[SampleLabelInterval] = []
        for j in np.flatnonzero(labeled):
            label = 1 if bool(long_label[j]) else 0
            anchor_index = int(a[j])
            ticks_observed = int(observed[j])
            feature_ts = int(ts[anchor_index])
            end_ts = int(ts[anchor_index + ticks_observed])
            sample_rows.append((feature_ts, Sample(tuple(float(x) for x in v[j]), label)))
            interval_rows.append(SampleLabelInterval(feature_ts, end_ts, ticks_observed))
        return sample_rows, interval_rows, skipped, int(long_label.sum()), int(short_label.sum())

    def build(self, store: ScalperStore, *, reset_learning: bool = True, progress=None) -> FastReplayReport:
        if reset_learning:
            store.reset_learning_state()
        ts, bid, ask = self._load_arrays(store)
        n = len(ts)
        if n < self.FEATURE_WINDOW + 10:
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

            # Existing helpers already commit in bounded batches. Keeping sample and interval
            # writes separate is safe because this is an offline research rebuild with execution off.
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
            # The fast builder starts from a learning reset by default. A mismatch would mean
            # callers deliberately disabled reset or the database contained colliding sample keys.
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
        )
