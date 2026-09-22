import tempfile
import unittest
from pathlib import Path

from truetrade.scalper import (
    AdaptiveSizer,
    DynamicProtectionManager,
    ExecutionQualityController,
    ForwardDemoQualifier,
    IntentKind,
    MicroFeatures,
    PositionState,
    RiskController,
    ScalperStore,
    Side,
    Tick,
)


def feature(**overrides):
    values = dict(
        ts_ns=10_000_000_000,
        mid=100.0,
        spread=0.20,
        spread_z=0.0,
        fast_velocity=0.8,
        slow_velocity=0.4,
        acceleration=0.4,
        tick_imbalance=0.5,
        trend_efficiency=0.7,
        volatility_ratio=0.8,
        last_move_ratio=0.2,
        noise_price=0.1,
    )
    values.update(overrides)
    return MicroFeatures(**values)


class ScalperPhase3Tests(unittest.TestCase):
    def test_dynamic_protection_tightens_only_after_favorable_excursion(self):
        manager = DynamicProtectionManager()
        position = PositionState(
            "p1", Side.LONG, 0.01, 100.0, 1,
            peak_exit_price=100.8, broker_stop=99.0,
        )
        intent = manager.evaluate(
            position,
            Tick(2, 100.7, 100.9),
            feature(mid=100.8),
            probability_long=0.78,
        )
        self.assertIsNotNone(intent)
        self.assertEqual(intent.kind, IntentKind.PROTECT)
        self.assertGreater(intent.stop, position.entry)
        self.assertGreater(intent.stop, position.broker_stop)

    def test_adaptive_sizer_blocks_weak_or_degraded_entries(self):
        sizer = AdaptiveSizer()
        risk = RiskController()
        strong = sizer.size(
            base_size=0.01, side=Side.LONG, confidence=0.90, threshold=0.62,
            features=feature(), positions=[], risk=risk,
            quality_multiplier=1.0, performance_multiplier=1.0, adding=False,
        )
        self.assertIsNotNone(strong)
        poor = sizer.size(
            base_size=0.01, side=Side.LONG, confidence=0.70, threshold=0.62,
            features=feature(volatility_ratio=3.0, spread_z=3.0), positions=[], risk=risk,
            quality_multiplier=0.20, performance_multiplier=1.0, adding=False,
        )
        self.assertIsNone(poor)

    def test_execution_quality_persists_and_can_block_new_entries(self):
        with tempfile.TemporaryDirectory() as td:
            store = ScalperStore(Path(td) / "state.sqlite")
            try:
                quality = ExecutionQualityController(store)
                for _ in range(12):
                    quality.observe_failure(uncertain=True)
                self.assertTrue(quality.blocked)
                restored = ExecutionQualityController(store)
                self.assertTrue(restored.blocked)
                self.assertGreater(restored.probability_penalty, 0)
                self.assertLess(restored.size_multiplier, 1)
            finally:
                store.close()

    def test_forward_demo_gate_requires_real_span_and_execution_quality(self):
        rows = []
        day = 86_400_000_000_000
        for i in range(120):
            pnl = 1.5 if i % 5 else -1.0
            rows.append(dict(
                position_id=str(i), decision_id=f"d{i}", broker_identifier=i + 1,
                model_generation=3, opened_ns=1 + i * day // 3,
                closed_ns=1 + i * day // 3 + 1_000_000,
                net_pnl=pnl, profit=pnl, commission=0.0, swap=0.0, fee=0.0,
                volume=0.01, risk_amount=1.0, entry_equity=10_000.0,
            ))
        gate = ForwardDemoQualifier()
        good = gate.evaluate(rows, 3, {"latency_p95_ms": 60.0, "slippage_p95_spreads": 0.20})
        self.assertTrue(good.eligible)
        self.assertGreater(good.expectancy_lower95_r, 0)
        bad = gate.evaluate(rows, 3, {"latency_p95_ms": 60.0, "slippage_p95_spreads": 2.0})
        self.assertFalse(bad.eligible)
        self.assertIn("forward_execution_slippage_unqualified", bad.reasons)

    def test_store_records_exact_model_generation_forward_outcome(self):
        with tempfile.TemporaryDirectory() as td:
            store = ScalperStore(Path(td) / "state.sqlite")
            try:
                self.assertTrue(store.create_execution_intent(
                    "d1", 10, kind="OPEN", position_id=None, side="LONG",
                    size=0.01, fraction=1.0, stop=None, detail="entry",
                    model_generation=4, entry_equity=1000.0,
                ))
                store.transition_execution_intent("d1", "submitted")
                store.transition_execution_intent(
                    "d1", "confirmed", broker_position_id="77",
                    broker_identifier=9001, risk_amount=2.0,
                )
                pending = store.confirmed_entry_intents_without_outcome()
                self.assertEqual(pending[0]["model_generation"], 4)
                outcome = dict(
                    position_id="77", decision_id="d1", broker_identifier=9001,
                    model_generation=4, opened_ns=10, closed_ns=20,
                    net_pnl=1.0, profit=1.0, commission=0.0, swap=0.0, fee=0.0,
                    volume=0.01, risk_amount=2.0, entry_equity=1000.0,
                )
                self.assertTrue(store.record_trade_outcome(outcome))
                self.assertEqual(len(store.trade_outcomes(4)), 1)
                self.assertEqual(store.confirmed_entry_intents_without_outcome(), [])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
