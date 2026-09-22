from __future__ import annotations

from dataclasses import dataclass, replace

from .features import TickFeatureEngine
from .labels import CostAwareLabeler
from .learning import Sample
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
            y = labeler.label(anchor, future)
            if y is None:
                skipped += 1
                continue
            labeled += int(store.add_sample(anchor.ts_ns, Sample(vector, y)))
        return ReplayReport(len(ticks), len(snapshots), labeled, skipped,
                            labeler.settings.extra_cost_spreads)
