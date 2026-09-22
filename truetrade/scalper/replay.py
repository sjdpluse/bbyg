from __future__ import annotations

from dataclasses import dataclass, replace

from .features import TickFeatureEngine
from .labels import CostAwareLabeler
from .learning import Sample
from .sample_intervals import clear_label_intervals, record_label_interval
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

        # A replay defines the complete derived sample set for the current tick history.
        # Keep interval evidence in sync with that derived state.
        clear_label_intervals(store)

        features = TickFeatureEngine()
        snapshots: list[tuple[int, object, tuple[float, ...]]] = []
        for idx, tick in enumerate(ticks):
            f = features.update(tick)
            if f is not None and idx % self.stride == 0:
                snapshots.append((idx, tick, f.vector()))

        labeled = 0
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
            inserted = store.add_sample(anchor.ts_ns, Sample(vector, outcome.label))
            if inserted:
                record_label_interval(
                    store,
                    anchor.ts_ns,
                    ticks[label_end_idx].ts_ns,
                    outcome.ticks_observed,
                )
                labeled += 1
        return ReplayReport(len(ticks), len(snapshots), labeled, skipped,
                            labeler.settings.extra_cost_spreads)
