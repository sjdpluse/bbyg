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
        path = [Tick(2, 99.7, 99.9), Tick(3, 100.5, 100.7)]
        outcome = self.labeler.outcome(anchor, path)
        self.assertNotEqual(outcome.label, 1)

    def test_later_recovery_after_short_stop_is_not_short_winner(self):
        anchor = Tick(1, 100.0, 100.2)
        path = [Tick(2, 100.5, 100.7), Tick(3, 99.5, 99.7)]
        outcome = self.labeler.outcome(anchor, path)
        self.assertNotEqual(outcome.label, 0)

    def test_entry_relative_stop_reports_actual_entry_risk(self):
        settings = LabelSettings(
            profit_spreads=1.6,
            loss_spreads=1.4,
            extra_cost_spreads=0.2,
            stop_reference="entry",
        )
        self.assertAlmostEqual(settings.nominal_target_from_entry_spreads, 1.8)
        self.assertAlmostEqual(settings.nominal_stop_from_entry_spreads, 1.4)
        spread = 0.2
        self.assertAlmostEqual(settings.long_stop_price(100.0, 100.2, spread), 99.92)
        self.assertAlmostEqual(settings.short_stop_price(100.0, 100.2, spread), 100.28)

    def test_fast_labels_enter_on_first_tick_after_decision(self):
        settings = LabelSettings(
            profit_spreads=1.0,
            loss_spreads=1.0,
            extra_cost_spreads=0.0,
            max_lookahead_ticks=10,
            stop_reference="entry",
        )
        builder = FastGapAwareReplayBuilder(labeler=CostAwareLabeler(settings), stride=1)

        # 96 warm-up ticks, then a feature/decision tick at index 96. Index 97 is the
        # executable entry and index 98 reaches the long target. If index 96 were used as
        # the entry, the recorded interval would be one tick shorter.
        bid = [99.9] * 96 + [100.0, 100.1, 100.5] + [100.5] * 10
        ask = [100.1] * 96 + [100.2, 100.3, 100.7] + [100.7] * 10
        ts = np.arange(1, len(bid) + 1, dtype=np.int64) * 1_000_000
        bid_arr = np.asarray(bid, dtype=float)
        ask_arr = np.asarray(ask, dtype=float)
        anchors = np.asarray([96], dtype=np.int64)
        vectors = np.zeros((1, 8), dtype=float)

        samples, intervals, skipped, long_count, short_count = builder._label_batch(
            anchors, vectors, ts, bid_arr, ask_arr, np.empty(0, dtype=np.int64)
        )
        self.assertEqual(skipped, 0)
        self.assertEqual(long_count, 1)
        self.assertEqual(short_count, 0)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0][1].y, 1)
        self.assertEqual(intervals[0].feature_ts_ns, int(ts[96]))
        self.assertEqual(intervals[0].label_end_ts_ns, int(ts[98]))
        self.assertEqual(intervals[0].ticks_observed, 2)

    def test_slow_next_tick_is_not_executable_entry(self):
        settings = LabelSettings(
            profit_spreads=1.0,
            loss_spreads=1.0,
            extra_cost_spreads=0.0,
            max_lookahead_ticks=10,
            max_entry_delay_seconds=0.5,
            stop_reference="entry",
        )
        builder = FastGapAwareReplayBuilder(labeler=CostAwareLabeler(settings), stride=1)
        bid = np.asarray([99.9] * 110, dtype=float)
        ask = np.asarray([100.1] * 110, dtype=float)
        ts = np.arange(110, dtype=np.int64) * 10_000_000
        ts[97:] += 1_000_000_000  # decision at 96 -> next tick delayed > 0.5s
        samples, intervals, skipped, _long, _short = builder._label_batch(
            np.asarray([96], dtype=np.int64),
            np.zeros((1, 8), dtype=float),
            ts,
            bid,
            ask,
            np.empty(0, dtype=np.int64),
        )
        self.assertEqual(samples, [])
        self.assertEqual(intervals, [])
        self.assertEqual(skipped, 1)


if __name__ == "__main__":
    unittest.main()
