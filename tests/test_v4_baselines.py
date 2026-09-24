import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from truetrade.scalper.experience import EpisodicMemory, MarketEpisode
from truetrade.scalper.store import ScalperStore
from truetrade.scalper.v4_baselines import BaselineSettings, run_v4_baselines


class V4BaselineTests(unittest.TestCase):
    def test_baseline_benchmark_runs_on_chronological_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ScalperStore(Path(tmp) / "scalper.sqlite")
            memory = EpisodicMemory(store)
            days = [
                "2026-09-16", "2026-09-17", "2026-09-18",
                "2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24",
            ]
            rng = np.random.default_rng(7)
            for day_i, day in enumerate(days):
                start = int(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() * 1e9)
                for i in range(90):
                    direction = 1 if (i + day_i) % 2 == 0 else -1
                    x = rng.normal(0, 0.15, size=47)
                    x[0] += 1.5 * direction
                    x[1] += 0.8 * direction
                    long_reward = 0.7 if direction > 0 else -0.65
                    short_reward = -0.65 if direction > 0 else 0.7
                    memory.add(MarketEpisode(
                        ts_ns=start + (i + 1) * 1_000_000_000,
                        state_embedding=tuple(float(v) for v in x),
                        regime="trend_up" if direction > 0 else "trend_down",
                        proposal="FLAT",
                        counterfactual_long_reward=long_reward,
                        counterfactual_short_reward=short_reward,
                    ))
            settings = BaselineSettings(
                minimum_training_samples=200,
                memory_max_prototypes=256,
                memory_k=8,
                linear_iterations=40,
            )
            result = run_v4_baselines(store, settings)
            self.assertFalse(result["execution_authorized"])
            self.assertEqual(result["counts"]["calibration"], 90)
            self.assertEqual(result["counts"]["diagnostic"], 90)
            self.assertIn("linear_logit", result["calibration"])
            self.assertIn("episodic_memory", result["calibration"])
            self.assertIn("linear_plus_memory", result["calibration"])
            self.assertGreater(
                result["calibration"]["linear_logit"]["classification"]["balanced_accuracy"],
                0.8,
            )
            self.assertEqual(
                result["pristine_validation"]["status"], "waiting_for_future_data"
            )
            store.close()


if __name__ == "__main__":
    unittest.main()
