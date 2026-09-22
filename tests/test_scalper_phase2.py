import tempfile
import unittest
from pathlib import Path

from truetrade.scalper import (
    ChampionChallenger, CostAwareLabeler, ExecutionTelemetry, LabelSettings,
    LearningSettings, Sample, ScalperStore, SelfImprovementController, Side, Tick,
    TickReplayBuilder,
)


class ScalperPhase2Tests(unittest.TestCase):
    def test_cost_aware_labels_are_event_based(self):
        labeler = CostAwareLabeler(LabelSettings(
            profit_spreads=1.0, loss_spreads=1.0, extra_cost_spreads=0.0,
            max_lookahead_ticks=20,
        ))
        anchor = Tick(1, 100.0, 100.2)
        unresolved = [Tick(i + 2, 100.05, 100.25) for i in range(20)]
        self.assertIsNone(labeler.label(anchor, unresolved))
        self.assertEqual(labeler.label(anchor, [Tick(2, 100.1, 100.3), Tick(3, 100.5, 100.7)]), 1)
        self.assertEqual(labeler.label(anchor, [Tick(2, 99.5, 99.7)]), 0)

    def test_tick_store_and_replay_are_durable(self):
        with tempfile.TemporaryDirectory() as td:
            store = ScalperStore(Path(td) / "scalper.sqlite")
            try:
                for i in range(900):
                    mid = 100 + i * 0.002 + (0.03 if i % 7 else -0.02)
                    store.append_tick(Tick((i + 1) * 1_000_000, mid - 0.05, mid + 0.05))
                report = TickReplayBuilder(
                    CostAwareLabeler(LabelSettings(
                        profit_spreads=0.5, loss_spreads=0.5,
                        extra_cost_spreads=0.0, max_lookahead_ticks=80,
                    )), stride=2,
                ).build(store)
                self.assertGreater(report.feature_rows, 100)
                self.assertGreater(store.sample_count(), 10)
            finally:
                store.close()

    def test_validation_block_is_consumed_once(self):
        with tempfile.TemporaryDirectory() as td:
            store = ScalperStore(Path(td) / "scalper.sqlite")
            learner = ChampionChallenger(8)
            try:
                for i in range(700):
                    y = i % 2
                    x = (0.0, 1.0 if y else -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    store.add_sample(i + 1, Sample(x, y))
                ctl = SelfImprovementController(
                    store, learner,
                    LearningSettings(min_train_samples=400, validation_block=120,
                                     purge_samples=20, min_logloss_improvement=0.001),
                )
                first = ctl.maybe_train()
                self.assertTrue(first.attempted)
                consumed = store.meta(ctl.META_LAST_VALIDATION)
                second = ctl.maybe_train()
                self.assertFalse(second.attempted)
                self.assertEqual(store.meta(ctl.META_LAST_VALIDATION), consumed)
            finally:
                store.close()

    def test_telemetry_measures_adverse_slippage(self):
        telemetry = ExecutionTelemetry()
        telemetry.record(start_ns=0, end_ns=2_000_000, expected_price=100.0,
                         fill_price=100.1, spread=0.2, side=Side.LONG, size=0.01)
        telemetry.record(start_ns=0, end_ns=4_000_000, expected_price=100.0,
                         fill_price=99.9, spread=0.2, side=Side.SHORT, size=0.01)
        summary = telemetry.summary()
        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["latency_p50_ms"], 3.0)
        self.assertGreater(summary["slippage_p50_spreads"], 0)


if __name__ == "__main__":
    unittest.main()
