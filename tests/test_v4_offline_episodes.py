import math
import tempfile
import unittest
from pathlib import Path

from truetrade.scalper.offline_episodes import OfflineEpisodeBuilder, OfflineEpisodeSettings
from truetrade.scalper.store import ScalperStore
from truetrade.scalper.types import Tick


class V4OfflineEpisodeTests(unittest.TestCase):
    @staticmethod
    def ticks(count=2200):
        base_ns = 1_790_000_000_000_000_000
        rows = []
        for i in range(count):
            trend = i * 0.0010
            wave = 0.22 * math.sin(i / 31.0) + 0.06 * math.sin(i / 7.0)
            mid = 4300.0 + trend + wave
            spread = 0.22 + 0.03 * (1.0 + math.sin(i / 47.0))
            rows.append(Tick(
                ts_ns=base_ns + i * 1_000_000_000,
                bid=mid - spread / 2,
                ask=mid + spread / 2,
                last=mid,
                volume=1.0 + (i % 11) * 0.05,
            ))
        return rows

    def test_build_persists_policy_independent_episodes_and_audits_exactly(self):
        settings = OfflineEpisodeSettings(
            stride=16,
            horizon_ticks=32,
            risk_spreads=2.0,
            extra_cost_spreads=0.20,
            max_gap_ns=300_000_000_000,
            outcome_chunk_size=128,
        )
        builder = OfflineEpisodeBuilder(settings)
        with tempfile.TemporaryDirectory() as tmp:
            store = ScalperStore(Path(tmp) / "scalper.sqlite")
            try:
                for tick in self.ticks():
                    store.append_tick(tick)
                report = builder.build(store, persist=True, replace=True)
                self.assertGreater(report.episodes_generated, 10)
                self.assertEqual(report.episodes_generated, report.episodes_persisted)
                self.assertGreater(report.feature_dimensions, 40)
                self.assertEqual(len(report.digest), 64)
                self.assertEqual(len(report.dataset_signature), 64)

                row = store.db.execute(
                    """SELECT proposal,executed,counterfactual_long_reward,counterfactual_short_reward,
                              context_json,state_embedding_json
                       FROM market_episodes ORDER BY id LIMIT 1"""
                ).fetchone()
                self.assertEqual(row[0], "FLAT")
                self.assertEqual(int(row[1]), 0)
                self.assertIsNotNone(row[2])
                self.assertIsNotNone(row[3])
                self.assertIn('"offline_policy_independent":true', row[4])
                self.assertTrue(row[5].startswith("["))

                audit = builder.parity_audit(store)
                self.assertTrue(audit["pass"])
                self.assertEqual(audit["expected_digest"], audit["actual_digest"])
                self.assertEqual(audit["expected_count"], audit["actual_count"])
            finally:
                store.close()

    def test_gap_creates_independent_segments_and_never_crosses_outcomes(self):
        ticks = self.ticks(2600)
        shift = 2_000_000_000_000
        split = 1300
        second = [
            Tick(t.ts_ns + shift, t.bid, t.ask, t.last, t.volume)
            for t in ticks[split:]
        ]
        combined = ticks[:split] + second
        settings = OfflineEpisodeSettings(
            stride=20,
            horizon_ticks=40,
            max_gap_ns=300_000_000_000,
            outcome_chunk_size=128,
        )
        builder = OfflineEpisodeBuilder(settings)
        with tempfile.TemporaryDirectory() as tmp:
            store = ScalperStore(Path(tmp) / "scalper.sqlite")
            try:
                report = builder.build(store, persist=False, ticks=combined)
                self.assertEqual(report.segments, 2)
                self.assertGreater(report.episodes_generated, 0)
            finally:
                store.close()

    def test_dataset_digest_changes_when_market_path_changes(self):
        settings = OfflineEpisodeSettings(
            stride=20,
            horizon_ticks=30,
            outcome_chunk_size=128,
        )
        builder = OfflineEpisodeBuilder(settings)
        original = self.ticks()
        changed = list(original)
        i = 1800
        t = changed[i]
        changed[i] = Tick(t.ts_ns, t.bid + 0.10, t.ask + 0.10, t.last + 0.10, t.volume)
        with tempfile.TemporaryDirectory() as tmp:
            store = ScalperStore(Path(tmp) / "scalper.sqlite")
            try:
                a = builder.build(store, persist=False, ticks=original)
                b = builder.build(store, persist=False, ticks=changed)
                self.assertEqual(a.dataset_signature, b.dataset_signature)
                self.assertNotEqual(a.digest, b.digest)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
