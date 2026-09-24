import tempfile
import unittest
from pathlib import Path

from truetrade.scalper.experience import (
    EpisodicMemory,
    MarketEpisode,
    RewardComponents,
    bounded_reward,
)
from truetrade.scalper.store import ScalperStore
from truetrade.scalper.thesis import (
    ThesisEvidence,
    ThesisSettings,
    ThesisState,
    TradeThesis,
)
from truetrade.scalper.types import Side


class V4MemoryTests(unittest.TestCase):
    def test_reward_is_bounded_and_penalizes_adverse_path(self):
        good = bounded_reward(RewardComponents(net_r=1.0, mfe_r=1.4, mae_r=0.2, execution_cost_r=0.05))
        bad = bounded_reward(RewardComponents(net_r=-1.0, mfe_r=0.1, mae_r=1.5, execution_cost_r=0.2))
        self.assertGreater(good, 0.0)
        self.assertLess(bad, 0.0)
        self.assertLessEqual(abs(good), 1.0)
        self.assertLessEqual(abs(bad), 1.0)

    def test_similarity_recall_prefers_matching_market_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ScalperStore(Path(tmp) / "state.sqlite")
            try:
                memory = EpisodicMemory(store)
                close_id = memory.add(MarketEpisode(
                    ts_ns=10,
                    state_embedding=(1.0, 0.0, 0.0),
                    regime="trend",
                    proposal="LONG",
                    executed=True,
                    realized_reward=0.7,
                    counterfactual_long_reward=0.8,
                    counterfactual_short_reward=-0.4,
                ))
                memory.add(MarketEpisode(
                    ts_ns=20,
                    state_embedding=(0.0, 1.0, 0.0),
                    regime="trend",
                    proposal="SHORT",
                    executed=True,
                    realized_reward=-0.5,
                    counterfactual_long_reward=-0.2,
                    counterfactual_short_reward=0.4,
                ))
                recalled = memory.recall((0.99, 0.01, 0.0), k=2, regime="trend")
                self.assertEqual(recalled.matches[0].episode_id, close_id)
                self.assertGreater(recalled.matches[0].similarity, 0.99)
                self.assertIsNotNone(recalled.long_minus_short_bias)
                self.assertGreater(recalled.long_minus_short_bias, 0.0)
            finally:
                store.close()

    def test_episode_outcome_can_be_resolved_later(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ScalperStore(Path(tmp) / "state.sqlite")
            try:
                memory = EpisodicMemory(store)
                episode_id = memory.add(MarketEpisode(
                    ts_ns=30,
                    state_embedding=(0.4, 0.3, 0.2),
                    regime="range",
                    proposal="FLAT",
                ))
                memory.update_outcome(
                    episode_id,
                    realized_reward=0.1,
                    counterfactual_long_reward=-0.3,
                    counterfactual_short_reward=-0.2,
                    mfe_r=0.4,
                    mae_r=0.1,
                    execution_cost_r=0.0,
                )
                episode = memory.get(episode_id)
                self.assertAlmostEqual(episode.realized_reward, 0.1)
                self.assertAlmostEqual(episode.counterfactual_long_reward, -0.3)
            finally:
                store.close()


class V4ThesisTests(unittest.TestCase):
    def setUp(self):
        self.settings = ThesisSettings(
            min_entry_observations=3,
            min_entry_age_seconds=1.0,
            min_entry_confidence=0.60,
            min_memory_confidence=0.20,
            min_directional_margin=0.05,
            min_exit_observations=3,
            min_exit_age_seconds=5.0,
        )

    def evidence(self, ts_ns, side=Side.LONG, *, model=0.68, tech=4, opposite=1,
                 trend=0.25, velocity=0.12, memory=0.7, bias=0.4):
        return ThesisEvidence(
            ts_ns=ts_ns,
            side=side,
            model_confidence=model,
            technical_score=tech,
            technical_opposite_score=opposite,
            trend_efficiency=trend if side is Side.LONG else -abs(trend),
            velocity=velocity if side is Side.LONG else -abs(velocity),
            memory_confidence=memory,
            memory_directional_bias=bias,
            novelty=0.2,
            regime_ok=True,
            fundamental_ok=True,
        )

    def test_entry_requires_persistent_evidence_not_one_tick(self):
        thesis = TradeThesis("t1", Side.LONG, 1_000_000_000, self.settings)
        self.assertEqual(thesis.observe_entry(self.evidence(1_100_000_000)), ThesisState.CANDIDATE)
        self.assertEqual(thesis.observe_entry(self.evidence(1_600_000_000)), ThesisState.CANDIDATE)
        self.assertEqual(thesis.observe_entry(self.evidence(2_100_000_000)), ThesisState.CONFIRMED)

    def test_exit_requires_age_and_repeated_reversal(self):
        thesis = TradeThesis("t2", Side.LONG, 1_000_000_000, self.settings)
        for ts in (1_100_000_000, 1_600_000_000, 2_100_000_000):
            thesis.observe_entry(self.evidence(ts))
        thesis.mark_entered(2_100_000_000)

        reverse = lambda ts: ThesisEvidence(
            ts_ns=ts,
            side=Side.SHORT,
            model_confidence=0.72,
            technical_score=4,
            technical_opposite_score=1,
            trend_efficiency=-0.3,
            velocity=-0.2,
            memory_confidence=0.8,
            memory_directional_bias=-0.5,
            novelty=0.2,
            regime_ok=True,
            fundamental_ok=True,
        )
        self.assertEqual(thesis.observe_exit(reverse(3_000_000_000)), ThesisState.ENTERED)
        self.assertEqual(thesis.observe_exit(reverse(4_000_000_000)), ThesisState.ENTERED)
        # Old enough now, but a third persistent reversal is still required.
        self.assertEqual(thesis.observe_exit(reverse(7_200_000_000)), ThesisState.EXITING)

    def test_memory_disagreement_blocks_entry(self):
        thesis = TradeThesis("t3", Side.LONG, 1_000_000_000, self.settings)
        conflicting = self.evidence(2_500_000_000, memory=0.9, bias=-0.4)
        for _ in range(3):
            thesis.observe_entry(conflicting)
        self.assertNotEqual(thesis.state, ThesisState.CONFIRMED)


if __name__ == "__main__":
    unittest.main()
