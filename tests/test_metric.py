import unittest

import numpy as np

from src.metric import mase, smape


class MetricTests(unittest.TestCase):
    def test_smape_is_zero_for_exact_forecast(self):
        values = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(smape(values, values), 0.0)

    def test_smape_handles_zero_values(self):
        actual = np.asarray([[0.0, 1.0]])
        predicted = np.asarray([[0.0, 1.0]])
        self.assertTrue(np.isfinite(smape(actual, predicted)))

    def test_mase_matches_manual_nonseasonal_scale(self):
        actual = np.asarray([[4.0, 5.0]])
        predicted = np.asarray([[3.0, 7.0]])
        insample = [np.asarray([1.0, 2.0, 3.0])]
        self.assertAlmostEqual(mase(actual, predicted, insample, seasonal_period=1), 1.5)


if __name__ == "__main__":
    unittest.main()
