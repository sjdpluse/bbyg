import math
import unittest

from truetrade.scalper import (
    AlgorithmicExitManager,
    ChampionChallenger,
    MicroFeatures,
    PositionState,
    RiskController,
    Sample,
    ScalperRiskLimits,
    Side,
    Tick,
    TickFeatureEngine,
)
from truetrade.scalper.types import IntentKind


def feature(**overrides):
    values = dict(
        ts_ns=10_000_000_000,
        mid=2000.0,
        spread=0.20,
        spread_z=0.0,
        fast_velocity=0.5,
        slow_velocity=0.3,
        acceleration=0.2,
        tick_imbalance=0.4,
        trend_efficiency=0.5,
        volatility_ratio=0.8,
        last_move_ratio=0.2,
        noise_price=0.1,
    )
    values.update(overrides)
    return MicroFeatures(**values)


class ScalperCoreTests(unittest.TestCase):
    def test_features_are_causal_finite_and_warm_up(self):
        e = TickFeatureEngine(min_ticks=24)
        out = None
        for i in range(24):
            out = e.update(Tick((i + 1) * 100_000_000, 2000 + i * 0.01, 2000.2 + i * 0.01))
        self.assertIsNotNone(out)
        self.assertTrue(all(math.isfinite(v) for v in out.vector()))
        self.assertGreater(out.fast_velocity, 0)

    def test_no_time_based_exit_when_edge_remains_positive(self):
        manager = AlgorithmicExitManager()
        pos = PositionState("p1", Side.LONG, 0.01, 2000.0, opened_ns=1)
        tick = Tick(10**18, 2000.7, 2000.9)
        result = manager.evaluate(pos, tick, feature(), probability_long=0.72)
        self.assertEqual(result.kind, IntentKind.HOLD)

    def test_edge_reversal_closes_position(self):
        manager = AlgorithmicExitManager()
        pos = PositionState("p1", Side.LONG, 0.01, 2000.0, opened_ns=1)
        tick = Tick(11_000_000_000, 1999.9, 2000.1)
        f = feature(fast_velocity=-0.5, tick_imbalance=-0.6, trend_efficiency=-0.4)
        result = manager.evaluate(pos, tick, f, probability_long=0.30)
        self.assertEqual(result.kind, IntentKind.CLOSE)
        self.assertEqual(result.reason, "edge_reversal")

    def test_profit_lock_is_market_state_based(self):
        manager = AlgorithmicExitManager()
        pos = PositionState("p1", Side.LONG, 0.01, 2000.0, opened_ns=1)
        manager.evaluate(pos, Tick(2_000_000_000, 2000.8, 2001.0), feature(), 0.75)
        result = manager.evaluate(
            pos,
            Tick(3_000_000_000, 2000.5, 2000.7),
            feature(fast_velocity=-0.1, trend_efficiency=0.0, tick_imbalance=-0.1),
            0.52,
        )
        self.assertEqual(result.kind, IntentKind.CLOSE)
        self.assertEqual(result.reason, "algorithmic_profit_lock")

    def test_risk_controller_allows_multiple_but_caps_inventory(self):
        r = RiskController(ScalperRiskLimits(max_positions=3, max_same_side_positions=2,
                                             max_total_size=0.03, max_directional_size=0.02,
                                             max_orders_per_second=5))
        positions = [
            PositionState("1", Side.LONG, 0.01, 2000, 1),
            PositionState("2", Side.LONG, 0.01, 2000, 1),
        ]
        ok, reason = r.can_open(positions, Side.LONG, 0.01, 0.0, now=1.0)
        self.assertFalse(ok)
        self.assertEqual(reason, "same_side_position_limit")
        ok, _ = r.can_open(positions, Side.SHORT, 0.01, 0.0, now=1.0)
        self.assertTrue(ok)

    def test_learning_requires_separate_validation_before_promotion(self):
        learner = ChampionChallenger(dimensions=8)
        base = learner.champion.w.copy()
        train = []
        validation = []
        for i in range(240):
            y = i % 2
            x = (0.0, 1.0 if y else -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            train.append(Sample(x, y))
        for i in range(120):
            y = i % 2
            x = (0.0, 1.0 if y else -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            validation.append(Sample(x, y))
        report = learner.fit_and_maybe_promote(train, validation)
        self.assertTrue(report.promoted)
        self.assertTrue(learner.qualified)
        self.assertFalse((learner.champion.w == base).all())

        frozen = learner.champion.w.copy()
        with self.assertRaises(ValueError):
            learner.fit_and_maybe_promote(train[:10], validation[:10])
        self.assertTrue((learner.champion.w == frozen).all())


if __name__ == "__main__":
    unittest.main()
