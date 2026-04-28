import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import os


class DataLoader:
    
    def __init__(self, data_dir: str = '../data'):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def fetch_stock_data(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = '1y'
    ) -> pd.DataFrame:
        
        if start_date is None and end_date is None:
            data = yf.download(
                tickers,
                period=period,
                auto_adjust=True,
                threads=True,
                progress=False
            )
        else:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                threads=True,
                progress=False
            )
        
        if len(tickers) == 1:
            data.columns = pd.MultiIndex.from_product([data.columns, tickers])
        
        return data
    
    def get_close_prices(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = '1y'
    ) -> pd.DataFrame:
        
        data = self.fetch_stock_data(tickers, start_date, end_date, period)
        
        if 'Close' in data.columns.get_level_values(0):
            close_prices = data['Close']
        else:
            close_prices = data
        
        close_prices = close_prices.dropna()
        
        return close_prices
    
    def get_returns(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = '1y',
        log_returns: bool = False
    ) -> pd.DataFrame:
        
        prices = self.get_close_prices(tickers, start_date, end_date, period)
        
        if log_returns:
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        
        returns = returns.dropna()
        
        return returns
    
    def get_stock_info(self, ticker: str) -> Dict:
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info
        except Exception as e:
            print(f"Error fetching info for {ticker}: {str(e)}")
            return {}
    
    def get_market_cap(self, ticker: str) -> Optional[float]:
        
        info = self.get_stock_info(ticker)
        return info.get('marketCap', None)
    
    def get_volume_data(
        self,
        tickers: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = '1y'
    ) -> pd.DataFrame:
        
        data = self.fetch_stock_data(tickers, start_date, end_date, period)
        
        if 'Volume' in data.columns.get_level_values(0):
            volume = data['Volume']
        else:
            volume = data
        
        volume = volume.dropna()
        
        return volume
    
    def save_data(self, data: pd.DataFrame, filename: str):
        
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath)
        print(f"Data saved to {filepath}")
    
    def load_data(self, filename: str) -> pd.DataFrame:
        
        filepath = os.path.join(self.data_dir, filename)
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return data
    
    def validate_tickers(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        
        valid_tickers = []
        invalid_tickers = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period='5d')
                if not hist.empty:
                    valid_tickers.append(ticker)
                else:
                    invalid_tickers.append(ticker)
            except:
                invalid_tickers.append(ticker)
        
        return valid_tickers, invalid_tickers
    
    def get_tadawul_ticker(self, stock_code: str) -> str:
        
        if not stock_code.endswith('.SR'):
            return f"{stock_code}.SR"
        return stock_code
    
    def fetch_multiple_periods(
        self,
        tickers: List[str],
        periods: List[str] = ['1mo', '3mo', '6mo', '1y', '2y']
    ) -> Dict[str, pd.DataFrame]:
        
        data_dict = {}
        for period in periods:
            data_dict[period] = self.get_close_prices(tickers, period=period)
        return data_dict
