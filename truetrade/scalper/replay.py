from __future__ import annotations

from dataclasses import dataclass, replace

from .features import TickFeatureEngine
from .labels import CostAwareLabeler
from .learning import Sample
from .sample_intervals import SampleLabelInterval, clear_label_intervals, record_label_intervals
from .store import ScalperStore


@dataclass(frozen=True)
class ReplayReport:
    ticks: int
    feature_rows: int
    labeled: int
    skipped: int
    extra_cost_spreads: float


class TickReplayBuilder:
    """Build causal samples; only the label builder may inspect future ticks."""

    def __init__(self, labeler: CostAwareLabeler | None = None, *, stride: int = 4):
        if stride < 1:
            raise ValueError("stride must be positive")
        self.labeler = labeler or CostAwareLabeler()
        self.stride = stride

    def build(self, store: ScalperStore, *, extra_cost_spreads: float | None = None) -> ReplayReport:
        ticks = store.ticks()
        labeler = self.labeler
        if extra_cost_spreads is not None:
            effective = max(labeler.settings.extra_cost_spreads, float(extra_cost_spreads))
            labeler = CostAwareLabeler(replace(labeler.settings, extra_cost_spreads=effective))

        clear_label_intervals(store)

        features = TickFeatureEngine()
        snapshots: list[tuple[int, object, tuple[float, ...]]] = []
        for idx, tick in enumerate(ticks):
            f = features.update(tick)
            if f is not None and idx % self.stride == 0:
                snapshots.append((idx, tick, f.vector()))

        sample_rows: list[tuple[int, Sample]] = []
        interval_rows: list[SampleLabelInterval] = []
        skipped = 0
        max_future = labeler.settings.max_lookahead_ticks
        for idx, anchor, vector in snapshots:
            future = ticks[idx + 1:idx + 1 + max_future]
            if len(future) < 10:
                skipped += 1
                continue
            outcome = labeler.outcome(anchor, future)
            if outcome.label is None:
                skipped += 1
                continue
            label_end_idx = idx + outcome.ticks_observed
            if label_end_idx >= len(ticks):
                skipped += 1
                continue
            sample_rows.append((anchor.ts_ns, Sample(vector, outcome.label)))
            interval_rows.append(
                SampleLabelInterval(anchor.ts_ns, ticks[label_end_idx].ts_ns, outcome.ticks_observed)
            )

        labeled = store.add_samples(sample_rows)
        # The replay is normally run after a learning reset. In case pre-existing samples
        # caused INSERT OR IGNORE collisions, only persist intervals for timestamps that
        # are now present in the sample table.
        if interval_rows:
            present = {r.feature_ts_ns for r in store.samples()}
            record_label_intervals(store, [r for r in interval_rows if r.feature_ts_ns in present])

        return ReplayReport(len(ticks), len(snapshots), labeled, skipped,
                            labeler.settings.extra_cost_spreads)
