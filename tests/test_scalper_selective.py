import unittest

import numpy as np

from truetrade.scalper.selective import choose_selective_threshold, selective_metrics


class ScalperSelectiveTests(unittest.TestCase):
    def test_selective_metrics_reports_coverage_and_accuracy(self):
        y = np.asarray([0, 0, 1, 1, 0, 1])
        p = np.asarray([0.10, 0.45, 0.90, 0.55, 0.20, 0.80])
        result = selective_metrics(y, p, 0.75)
        self.assertEqual(result.selected, 4)
        self.assertAlmostEqual(result.coverage, 4 / 6)
        self.assertAlmostEqual(result.accuracy, 1.0)
        self.assertAlmostEqual(result.balanced_accuracy, 1.0)

    def test_threshold_selection_respects_minimum_evidence(self):
        y = np.asarray([0, 1] * 100)
        p = np.asarray([0.2, 0.8] * 100, dtype=float)
        threshold, rows = choose_selective_threshold(
            y, p, min_coverage=0.25, min_selected=50, min_class_count=20
        )
        self.assertIsNotNone(threshold)
        self.assertTrue(any(row["eligible"] for row in rows))

    def test_threshold_selection_fails_closed_with_no_usable_signals(self):
        y = np.asarray([0, 1] * 50)
        p = np.full(100, 0.5)
        threshold, rows = choose_selective_threshold(
            y, p, min_coverage=0.10, min_selected=10, min_class_count=5
        )
        self.assertIsNone(threshold)
        self.assertFalse(any(row["eligible"] for row in rows))


if __name__ == "__main__":
    unittest.main()
