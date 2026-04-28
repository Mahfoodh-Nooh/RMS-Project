import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import pandas as pd
from src.portfolio import Portfolio


class TestPortfolio(unittest.TestCase):
    
    def setUp(self):
        self.tickers = ['AAPL', 'MSFT', 'GOOGL']
        self.initial_value = 1000000
        self.weights = np.array([0.4, 0.3, 0.3])
        
        self.portfolio = Portfolio(
            name="Test Portfolio",
            initial_value=self.initial_value,
            tickers=self.tickers,
            weights=self.weights
        )
    
    def test_initialization(self):
        self.assertEqual(self.portfolio.name, "Test Portfolio")
        self.assertEqual(self.portfolio.initial_value, self.initial_value)
        self.assertEqual(len(self.portfolio.tickers), 3)
        np.testing.assert_array_almost_equal(self.portfolio.weights, self.weights)
    
    def test_allocation(self):
        for ticker in self.tickers:
            self.assertIn(ticker, self.portfolio.allocation)
            self.assertIn('weight', self.portfolio.allocation[ticker])
            self.assertIn('value', self.portfolio.allocation[ticker])
    
    def test_update_prices(self):
        current_prices = pd.Series({
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 2500.0
        })
        
        self.portfolio.update_prices(current_prices)
        
        for ticker in self.tickers:
            self.assertIn('current_value', self.portfolio.allocation[ticker])
            self.assertIn('shares', self.portfolio.allocation[ticker])
    
    def test_add_cash(self):
        initial = self.portfolio.current_value
        self.portfolio.add_cash(100000)
        self.assertEqual(self.portfolio.current_value, initial + 100000)
    
    def test_withdraw_cash(self):
        initial = self.portfolio.current_value
        success = self.portfolio.withdraw_cash(50000)
        self.assertTrue(success)
        self.assertEqual(self.portfolio.current_value, initial - 50000)
    
    def test_withdraw_cash_insufficient(self):
        success = self.portfolio.withdraw_cash(2000000)
        self.assertFalse(success)
    
    def test_export_import(self):
        portfolio_data = self.portfolio.export_portfolio()
        
        imported_portfolio = Portfolio.import_portfolio(portfolio_data)
        
        self.assertEqual(imported_portfolio.name, self.portfolio.name)
        self.assertEqual(imported_portfolio.initial_value, self.portfolio.initial_value)
        np.testing.assert_array_equal(imported_portfolio.weights, self.portfolio.weights)


if __name__ == '__main__':
    unittest.main()
