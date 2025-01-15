import unittest
import numpy as np
from src.metrics.classification import calculate_brier_score

class TestMetrics(unittest.TestCase):
    def test_calculate_brier_score(self):
        predictions = np.array([0.1, 0.4, 0.35, 0.8])
        labels = np.array([0, 0, 1, 1])
        brier_score = calculate_brier_score(predictions, labels)
        self.assertAlmostEqual(brier_score, 0.1427, places=4)

if __name__ == '__main__':
    unittest.main()