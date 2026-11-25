"""Small scientific-boundary tests; no GPU or reference data required."""
import unittest
import numpy as np
from run import direction_metrics


class DirectionMetricsTest(unittest.TestCase):
    def test_abstention_stays_in_denominator_and_experimental_zero_excluded(self):
        m = direction_metrics([1, -2, 1e-8, 9], [2, -3, 1, 0], 1e-6)
        self.assertEqual((m['n_genes'], m['n_correct'], m['n_abstain']), (3, 2, 1))
        self.assertAlmostEqual(m['accuracy'], 2/3)
        self.assertEqual(m['accuracy_called_only'], 1)
        self.assertEqual(m['n_experimental_zero'], 1)
        self.assertAlmostEqual(m['balanced_accuracy'], .75)

    def test_effect_sign_is_not_discarded(self):
        m = direction_metrics([-1, -2, -3, -4], [1, 2, 3, 4], 1e-6)
        self.assertEqual(m['accuracy'], 0)
        self.assertAlmostEqual(m['spearman_signed_effect'], -1)
        self.assertAlmostEqual(m['pearson_signed_effect'], -1)
        self.assertEqual(m['majority_accuracy'], 1)
        self.assertTrue(np.isnan(m['balanced_accuracy']))

    def test_empty_and_nonfinite(self):
        self.assertTrue(np.isnan(direction_metrics([], [], 1e-6)['accuracy']))
        with self.assertRaises(ValueError):
            direction_metrics([np.nan], [1], 1e-6)


if __name__ == '__main__':
    unittest.main()
