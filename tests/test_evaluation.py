import unittest
import numpy as np
import pandas as pd
from src.evaluation.evaluate import bootstrap_metrics

class TestEvaluation(unittest.TestCase):
    def test_bootstrap_metrics(self):
        y_test = pd.Series([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.4, 0.35, 0.8])
        threshold = 0.5
        results = bootstrap_metrics(y_test, y_prob, threshold, n_bootstraps=10)
        self.assertIn('AUROC', results)

if __name__ == '__main__':
    unittest.main()