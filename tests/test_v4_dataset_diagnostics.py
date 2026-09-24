import json
import tempfile
import unittest
from pathlib import Path

from truetrade.scalper.dataset_diagnostics import V4DatasetDiagnostics
from truetrade.scalper.experience import EpisodicMemory, MarketEpisode
from truetrade.scalper.store import ScalperStore


class V4DatasetDiagnosticsTests(unittest.TestCase):
    def test_profiles_rewards_and_builds_day_split(self):
        base = 1_790_000_000_000_000_000
        day_ns = 86_400_000_000_000
        with tempfile.TemporaryDirectory() as tmp:
            with ScalperStore(Path(tmp) / "scalper.sqlite") as store:
                memory = EpisodicMemory(store)
                for day in range(4):
                    for i in range(120):
                        ts = base + day * day_ns + i * 1_000_000_000
                        long_reward = 0.20 if i % 2 == 0 else -0.10
                        short_reward = -0.10 if i % 2 == 0 else 0.20
                        memory.add(MarketEpisode(
                            ts_ns=ts,
                            state_embedding=(1.0, float(i + 1), 0.5),
                            regime="trend_up" if i % 3 else "range",
                            proposal="FLAT",
                            counterfactual_long_reward=long_reward,
                            counterfactual_short_reward=short_reward,
                            context={"long_net_r": long_reward, "short_net_r": short_reward},
                        ))
                report = V4DatasetDiagnostics(min_day_episodes=100).analyze(store)
        self.assertEqual(report.episode_count, 480)
        self.assertEqual(report.feature_dimensions, 3)
        self.assertEqual(report.day_count, 4)
        self.assertIsNotNone(report.split_plan)
        self.assertEqual(report.split_plan.calibration_count, 120)
        self.assertEqual(report.split_plan.validation_count, 120)
        self.assertAlmostEqual(report.directional_edge["long_better_fraction"], 0.5)
        self.assertAlmostEqual(report.directional_edge["short_better_fraction"], 0.5)

    def test_missing_days_prevents_split(self):
        base = 1_790_000_000_000_000_000
        with tempfile.TemporaryDirectory() as tmp:
            with ScalperStore(Path(tmp) / "scalper.sqlite") as store:
                memory = EpisodicMemory(store)
                for i in range(120):
                    memory.add(MarketEpisode(
                        ts_ns=base + i * 1_000_000_000,
                        state_embedding=(1.0, float(i + 1)),
                        regime="range",
                        proposal="FLAT",
                        counterfactual_long_reward=0.1,
                        counterfactual_short_reward=-0.1,
                        context={"long_net_r": 0.1, "short_net_r": -0.1},
                    ))
                report = V4DatasetDiagnostics(min_day_episodes=100).analyze(store)
        self.assertIsNone(report.split_plan)
        self.assertFalse(report.gates["split_possible"])


if __name__ == "__main__":
    unittest.main()
