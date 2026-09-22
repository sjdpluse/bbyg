import unittest

from truetrade.scalper.economic_gate import EconomicThresholdCandidate, choose_economic_threshold


class EconomicThresholdGateTests(unittest.TestCase):
    def test_gate_abstains_when_no_threshold_has_positive_expectancy(self):
        rows = [
            EconomicThresholdCandidate(0.53, 400, -12.0, -0.03, 0.91, 0.54),
            EconomicThresholdCandidate(0.57, 120, -1.0, -0.01, 0.99, 0.58),
        ]
        self.assertIsNone(choose_economic_threshold(rows, min_trades=100, min_profit_factor=1.05))

    def test_gate_ignores_tiny_profitable_sample(self):
        rows = [
            EconomicThresholdCandidate(0.59, 20, 8.0, 0.4, 1.8, 0.7),
            EconomicThresholdCandidate(0.55, 150, 5.0, 0.03, 1.08, 0.56),
        ]
        chosen = choose_economic_threshold(rows, min_trades=100, min_profit_factor=1.05)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.threshold, 0.55)

    def test_gate_prefers_total_calibration_expectancy_among_eligible_thresholds(self):
        rows = [
            EconomicThresholdCandidate(0.54, 300, 12.0, 0.04, 1.10, 0.56),
            EconomicThresholdCandidate(0.57, 140, 18.0, 0.13, 1.24, 0.61),
        ]
        chosen = choose_economic_threshold(rows, min_trades=100, min_profit_factor=1.05)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.threshold, 0.57)


if __name__ == "__main__":
    unittest.main()
