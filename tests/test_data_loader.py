import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
import pandas as pd
from src.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    
    def setUp(self):
        self.loader = DataLoader(data_dir='../data')
        self.test_tickers = ['AAPL', 'MSFT']
    
    def test_fetch_stock_data(self):
        data = self.loader.fetch_stock_data(self.test_tickers, period='1mo')
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(data.empty)
    
    def test_get_close_prices(self):
        prices = self.loader.get_close_prices(self.test_tickers, period='1mo')
        self.assertIsInstance(prices, pd.DataFrame)
        self.assertEqual(len(prices.columns), len(self.test_tickers))
    
    def test_get_returns(self):
        returns = self.loader.get_returns(self.test_tickers, period='1mo')
        self.assertIsInstance(returns, pd.DataFrame)
        self.assertFalse(returns.isnull().all().any())
    
    def test_tadawul_ticker_conversion(self):
        ticker = self.loader.get_tadawul_ticker('2222')
        self.assertEqual(ticker, '2222.SR')
        
        ticker = self.loader.get_tadawul_ticker('2222.SR')
        self.assertEqual(ticker, '2222.SR')


if __name__ == '__main__':
    unittest.main()
