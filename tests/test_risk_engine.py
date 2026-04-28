import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import pandas as pd
from src.risk_engine import RiskEngine


class TestRiskEngine(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        self.returns = pd.DataFrame({
            'Asset1': np.random.normal(0.001, 0.02, 252),
            'Asset2': np.random.normal(0.0008, 0.015, 252),
            'Asset3': np.random.normal(0.0012, 0.018, 252)
        }, index=dates)
        
        self.risk_engine = RiskEngine(self.returns, confidence_levels=[0.95, 0.99])
        self.portfolio_value = 1000000
        self.weights = np.array([0.4, 0.3, 0.3])
    
    def test_calculate_returns_stats(self):
        stats = self.risk_engine.calculate_returns_stats()
        self.assertIn('mean_return', stats)
        self.assertIn('std_dev', stats)
        self.assertIn('annualized_return', stats)
        self.assertIn('annualized_volatility', stats)
    
    def test_calculate_correlation_matrix(self):
        corr_matrix = self.risk_engine.calculate_correlation_matrix()
        self.assertIsInstance(corr_matrix, pd.DataFrame)
        self.assertEqual(corr_matrix.shape, (3, 3))
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), [1.0, 1.0, 1.0])
    
    def test_calculate_parametric_var(self):
        var = self.risk_engine.calculate_parametric_var(self.portfolio_value, self.weights)
        self.assertIn(0.95, var)
        self.assertIn(0.99, var)
        self.assertTrue(var[0.99] >= var[0.95])
    
    def test_calculate_historical_var(self):
        var = self.risk_engine.calculate_historical_var(self.portfolio_value, self.weights)
        self.assertIn(0.95, var)
        self.assertIn(0.99, var)
        self.assertTrue(all(v > 0 for v in var.values()))
    
    def test_calculate_cvar(self):
        cvar = self.risk_engine.calculate_cvar(self.portfolio_value, self.weights)
        self.assertIn(0.95, cvar)
        self.assertIn(0.99, cvar)
    
    def test_calculate_sharpe_ratio(self):
        sharpe = self.risk_engine.calculate_sharpe_ratio(self.weights)
        self.assertIsInstance(sharpe, (float, np.floating))
    
    def test_monte_carlo_simulation(self):
        mc_results = self.risk_engine.monte_carlo_simulation(
            portfolio_value=self.portfolio_value,
            weights=self.weights,
            num_simulations=1000,
            time_horizon=100
        )
        self.assertIn('simulated_paths', mc_results)
        self.assertIn('final_values', mc_results)
        self.assertIn('mc_var', mc_results)
        self.assertEqual(len(mc_results['final_values']), 1000)
    
    def test_stress_test(self):
        stress_results = self.risk_engine.stress_test(self.portfolio_value, self.weights)
        self.assertIn('moderate_shock', stress_results)
        self.assertIn('severe_shock', stress_results)


if __name__ == '__main__':
    unittest.main()
