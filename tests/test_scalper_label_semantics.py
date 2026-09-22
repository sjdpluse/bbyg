import unittest

import numpy as np

from truetrade.scalper.fast_replay import FastGapAwareReplayBuilder
from truetrade.scalper.labels import CostAwareLabeler, LabelSettings
from truetrade.scalper.types import Tick


class StopAwareLabelTests(unittest.TestCase):
    def setUp(self):
        self.labeler = CostAwareLabeler(LabelSettings(
            profit_spreads=1.0,
            loss_spreads=1.0,
            extra_cost_spreads=0.0,
            max_lookahead_ticks=20,
        ))

    def test_later_recovery_after_long_stop_is_not_long_winner(self):
        anchor = Tick(1, 100.0, 100.2)
        # Long stop is 99.8 and long target is 100.4 executable bid.  The long
        # hypothesis stops first, then price later rallies through its old target.
        path = [
            Tick(2, 99.7, 99.9),
            Tick(3, 100.5, 100.7),
        ]
        outcome = self.labeler.outcome(anchor, path)
        self.assertNotEqual(outcome.label, 1)

    def test_later_recovery_after_short_stop_is_not_short_winner(self):
        anchor = Tick(1, 100.0, 100.2)
        # Short stop is 100.4 and short target is 99.8 executable ask.
        path = [
            Tick(2, 100.5, 100.7),
            Tick(3, 99.5, 99.7),
        ]
        outcome = self.labeler.outcome(anchor, path)
        self.assertNotEqual(outcome.label, 0)

    def test_fast_first_passage_decision_matches_scalar_for_known_paths(self):
        settings = LabelSettings(
            profit_spreads=1.0,
            loss_spreads=1.0,
            extra_cost_spreads=0.0,
            max_lookahead_ticks=10,
        )
        labeler = CostAwareLabeler(settings)
        builder = FastGapAwareReplayBuilder(labeler=labeler, stride=1)

        # Include 96 warm-up ticks, then three anchor trajectories.  We call the
        # vectorized label helper directly so this test is about label semantics only.
        bid = [99.9] * 96
        ask = [100.1] * 96
        ts = list(range(1, 97))

        # Anchor A: long target before its stop -> LONG.
        bid += [100.0, 100.5, 100.5, 100.5]
        ask += [100.2, 100.7, 100.7, 100.7]
        ts += list(range(97, 101))
        # Anchor B: short target before its stop -> SHORT.
        bid += [100.0, 99.5, 99.5, 99.5]
        ask += [100.2, 99.7, 99.7, 99.7]
        ts += list(range(101, 105))
        # Anchor C: long stops, then its old target is reached; must not label LONG.
        bid += [100.0, 99.7, 100.5, 100.5]
        ask += [100.2, 99.9, 100.7, 100.7]
        ts += list(range(105, 109))

        ts_arr = np.asarray(ts, dtype=np.int64)
        bid_arr = np.asarray(bid, dtype=float)
        ask_arr = np.asarray(ask, dtype=float)
        anchors = np.asarray([96, 100, 104], dtype=np.int64)
        vectors = np.zeros((3, 8), dtype=float)
        samples, _intervals, _skipped, _long, _short = builder._label_batch(
            anchors,
            vectors,
            ts_arr,
            bid_arr,
            ask_arr,
            np.empty(0, dtype=np.int64),
        )
        labels = {feature_ts: sample.y for feature_ts, sample in samples}
        self.assertEqual(labels.get(int(ts_arr[96])), 1)
        self.assertEqual(labels.get(int(ts_arr[100])), 0)
        self.assertNotEqual(labels.get(int(ts_arr[104])), 1)


if __name__ == "__main__":
    unittest.main()
