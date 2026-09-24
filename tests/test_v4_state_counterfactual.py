import math
import unittest

from truetrade.scalper.counterfactual import CounterfactualEpisodeResolver, CounterfactualSettings
from truetrade.scalper.features import TickFeatureEngine
from truetrade.scalper.state_encoder import MarketStateEncoder
from truetrade.scalper.types import Tick


class V4StateCounterfactualTests(unittest.TestCase):
    @staticmethod
    def ticks(count=1200):
        base_ns = 1_790_000_000_000_000_000
        rows = []
        for i in range(count):
            trend = i * 0.0008
            wave = 0.18 * math.sin(i / 23.0) + 0.05 * math.sin(i / 5.0)
            mid = 4300.0 + trend + wave
            spread = 0.24 + 0.02 * (1.0 + math.sin(i / 31.0))
            rows.append(Tick(
                ts_ns=base_ns + i * 1_000_000_000,
                bid=mid - spread / 2,
                ask=mid + spread / 2,
                last=mid,
                volume=1.0 + (i % 7) * 0.1,
            ))
        return rows

    def test_state_encoder_is_deterministic_and_multiscale(self):
        ticks = self.ticks()
        outputs = []
        for _ in range(2):
            micro = TickFeatureEngine(window=96, min_ticks=96, fast_ticks=8, slow_ticks=24)
            encoder = MarketStateEncoder()
            last = None
            for tick in ticks:
                last = encoder.update(tick, micro.update(tick)) or last
            self.assertIsNotNone(last)
            self.assertEqual(len(last.embedding), encoder.dimensions)
            self.assertGreater(encoder.dimensions, 40)
            self.assertIn(last.regime, {"quiet", "range", "trend_up", "trend_down", "shock"})
            outputs.append(last)
        self.assertEqual(outputs[0].feature_names, outputs[1].feature_names)
        self.assertEqual(outputs[0].regime, outputs[1].regime)
        self.assertEqual(outputs[0].embedding, outputs[1].embedding)

    def test_encoder_rejects_nonchronological_ticks(self):
        encoder = MarketStateEncoder()
        tick = self.ticks(1)[0]
        encoder.update(tick, None)
        with self.assertRaises(ValueError):
            encoder.update(tick, None)

    def test_counterfactual_resolves_both_directions_from_same_path(self):
        ticks = self.ticks(30)
        resolver = CounterfactualEpisodeResolver(CounterfactualSettings(
            horizon_ticks=12,
            risk_spreads=2.0,
            extra_cost_spreads=0.20,
        ))
        resolver.add(ticks[0])
        resolved = []
        for tick in ticks[1:]:
            resolved.extend(resolver.advance(tick))
            if resolved:
                break
        self.assertEqual(len(resolved), 1)
        outcome = resolved[0]
        self.assertEqual(outcome.anchor_ts_ns, ticks[0].ts_ns)
        self.assertEqual(outcome.ticks_observed, 12)
        self.assertTrue(-1.0 <= outcome.long.reward <= 1.0)
        self.assertTrue(-1.0 <= outcome.short.reward <= 1.0)
        self.assertGreaterEqual(outcome.long.mfe_r, 0.0)
        self.assertGreaterEqual(outcome.short.mfe_r, 0.0)
        self.assertGreaterEqual(outcome.long.mae_r, 0.0)
        self.assertGreaterEqual(outcome.short.mae_r, 0.0)
        self.assertEqual(resolver.pending_count(), 0)

    def test_counterfactual_never_uses_anchor_as_entry(self):
        base = 1_790_000_000_000_000_000
        anchor = Tick(base, 100.0, 100.2, 100.1, 1.0)
        future = [
            Tick(base + i * 1_000_000_000, 101.0 + i, 101.2 + i, 101.1 + i, 1.0)
            for i in range(1, 13)
        ]
        resolver = CounterfactualEpisodeResolver(CounterfactualSettings(horizon_ticks=10))
        resolver.add(anchor)
        result = []
        for tick in future:
            result.extend(resolver.advance(tick))
            if result:
                break
        self.assertEqual(len(result), 1)
        # Entry must be the first strictly later ask/bid, never the anchor's quote.
        self.assertAlmostEqual(result[0].long.entry_price, future[0].ask)
        self.assertAlmostEqual(result[0].short.entry_price, future[0].bid)


if __name__ == "__main__":
    unittest.main()
