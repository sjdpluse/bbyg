import unittest

import numpy as np

from truetrade.scalper.economic_replay import (
    CostScenario,
    EconomicPolicy,
    economic_metrics,
    simulate_selective_trades,
)
from truetrade.scalper.labels import LabelSettings


class EconomicReplayTests(unittest.TestCase):
    def setUp(self):
        self.policy = EconomicPolicy(
            "test",
            max_positions=2,
            max_same_side_positions=2,
            max_entries_per_second=10,
            cooldown_ms=0,
            max_entry_delay_seconds=5.0,
        )
        self.labels = LabelSettings(
            profit_spreads=1.0,
            loss_spreads=1.0,
            extra_cost_spreads=0.0,
            max_lookahead_ticks=20,
        )

    def test_signal_enters_on_next_tick_and_target_uses_executable_bid(self):
        ts = np.asarray([1, 2, 3, 4], dtype=np.int64) * 1_000_000_000
        bid = np.asarray([100.0, 100.0, 100.5, 100.5])
        ask = np.asarray([100.2, 100.2, 100.7, 100.7])
        trades, flow = simulate_selective_trades(
            tick_ts=ts,
            bid=bid,
            ask=ask,
            signal_ts=np.asarray([ts[0]]),
            probability_long=np.asarray([0.60]),
            threshold=0.55,
            policy=self.policy,
            label_settings=self.labels,
        )
        self.assertEqual(flow["entries_opened"], 1)
        self.assertEqual(len(trades), 1)
        trade = trades[0]
        self.assertEqual(trade.entry_ts_ns, int(ts[1]))
        self.assertEqual(trade.entry, float(ask[1]))
        self.assertEqual(trade.exit_reason, "target")
        self.assertAlmostEqual(trade.gross_pnl_spreads, 1.0)

    def test_stopped_trade_does_not_recover_into_winner(self):
        ts = np.asarray([1, 2, 3, 4], dtype=np.int64) * 1_000_000_000
        bid = np.asarray([100.0, 100.0, 99.7, 100.6])
        ask = np.asarray([100.2, 100.2, 99.9, 100.8])
        trades, _ = simulate_selective_trades(
            tick_ts=ts,
            bid=bid,
            ask=ask,
            signal_ts=np.asarray([ts[0]]),
            probability_long=np.asarray([0.60]),
            threshold=0.55,
            policy=self.policy,
            label_settings=self.labels,
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].exit_reason, "stop")
        self.assertLess(trades[0].gross_pnl_spreads, 0.0)

    def test_cost_scenarios_reduce_same_gross_trade_deterministically(self):
        ts = np.asarray([1, 2, 3, 4], dtype=np.int64) * 1_000_000_000
        bid = np.asarray([100.0, 100.0, 100.5, 100.5])
        ask = np.asarray([100.2, 100.2, 100.7, 100.7])
        trades, _ = simulate_selective_trades(
            tick_ts=ts,
            bid=bid,
            ask=ask,
            signal_ts=np.asarray([ts[0]]),
            probability_long=np.asarray([0.60]),
            threshold=0.55,
            policy=self.policy,
            label_settings=self.labels,
        )
        gross = economic_metrics(trades, CostScenario("gross", 0.0, 0.0))
        stressed = economic_metrics(trades, CostScenario("stress", 0.10, 0.20))
        self.assertAlmostEqual(gross["net_pnl_spreads"], 1.0)
        self.assertAlmostEqual(stressed["cost_per_trade_spreads"], 0.40)
        self.assertAlmostEqual(stressed["net_pnl_spreads"], 0.60)

    def test_single_position_policy_blocks_overlap(self):
        ts = np.asarray([1, 2, 3, 4, 5], dtype=np.int64) * 1_000_000_000
        bid = np.asarray([100.0, 100.0, 100.0, 100.0, 100.5])
        ask = np.asarray([100.2, 100.2, 100.2, 100.2, 100.7])
        policy = EconomicPolicy(
            "single", max_positions=1, max_same_side_positions=1,
            max_entries_per_second=10, cooldown_ms=0,
        )
        _trades, flow = simulate_selective_trades(
            tick_ts=ts,
            bid=bid,
            ask=ask,
            signal_ts=np.asarray([ts[0], ts[1]]),
            probability_long=np.asarray([0.60, 0.60]),
            threshold=0.55,
            policy=policy,
            label_settings=self.labels,
        )
        self.assertEqual(flow["entries_opened"], 1)
        self.assertEqual(flow["entries_blocked_position_limit"], 1)


if __name__ == "__main__":
    unittest.main()
