import tempfile
import unittest
from pathlib import Path

import numpy as np

from truetrade.scalper.fast_replay import FastGapAwareReplayBuilder
from truetrade.scalper.features import TickFeatureEngine
from truetrade.scalper.store import ScalperStore
from truetrade.scalper.types import Tick


class FastReplayTests(unittest.TestCase):
    def test_vectorized_features_match_live_engine_after_full_warmup(self):
        ticks = []
        for i in range(180):
            mid = 2000.0 + 0.03 * np.sin(i / 5.0) + 0.002 * i
            spread = 0.20 + 0.01 * (i % 3)
            ticks.append(Tick(1_000_000_000 + i * 10_000_000, mid - spread / 2, mid + spread / 2))

        engine = TickFeatureEngine()
        expected = {}
        for i, tick in enumerate(ticks):
            features = engine.update(tick)
            if features is not None and i >= 95 and i % 4 == 0:
                expected[i] = np.asarray(features.vector())

        anchors = np.asarray(sorted(expected), dtype=np.int64)
        bid = np.asarray([t.bid for t in ticks])
        ask = np.asarray([t.ask for t in ticks])
        actual = FastGapAwareReplayBuilder._feature_matrix(anchors, bid, ask)
        for row, anchor in zip(actual, anchors):
            np.testing.assert_allclose(row, expected[int(anchor)], rtol=1e-11, atol=1e-11)

    def test_label_intervals_never_cross_market_gap(self):
        with tempfile.TemporaryDirectory() as td:
            store = ScalperStore(Path(td) / "state.sqlite")
            try:
                ticks = []
                ts = 1_000_000_000
                price = 2000.0
                for segment in range(2):
                    for i in range(180):
                        price += 0.03 if i % 7 < 4 else -0.02
                        ticks.append(Tick(ts, price - 0.10, price + 0.10))
                        ts += 10_000_000
                    if segment == 0:
                        ts += 3_600_000_000_000
                with store.db:
                    store.db.executemany(
                        "INSERT INTO ticks(ts_ns,bid,ask,last,volume) VALUES(?,?,?,?,?)",
                        [(t.ts_ns, t.bid, t.ask, t.last, t.volume) for t in ticks],
                    )

                report = FastGapAwareReplayBuilder(stride=4, max_gap_seconds=300).build(store)
                self.assertEqual(report.market_gaps, 1)
                gap_start = ticks[180].ts_ns
                crossings = store.db.execute(
                    """SELECT count(*) FROM sample_label_intervals
                       WHERE feature_ts_ns < ? AND label_end_ts_ns >= ?""",
                    (gap_start, gap_start),
                ).fetchone()[0]
                self.assertEqual(crossings, 0)
                self.assertEqual(
                    store.sample_count(),
                    store.db.execute("SELECT count(*) FROM sample_label_intervals").fetchone()[0],
                )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
