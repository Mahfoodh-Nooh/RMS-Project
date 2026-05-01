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
        
        kwargs = dict(auto_adjust=True, progress=False)
        if start_date is None and end_date is None:
            kwargs['period'] = period
        else:
            kwargs['start'] = start_date
            kwargs['end'] = end_date

        if len(tickers) == 1:
            data = yf.download(tickers[0], **kwargs)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    pass
                else:
                    data.columns = pd.MultiIndex.from_product([data.columns, tickers])
        else:
            data = yf.download(tickers, **kwargs)
            if not data.empty and not isinstance(data.columns, pd.MultiIndex):
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

        if data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                close_prices = data['Close']
            elif 'Adj Close' in data.columns.get_level_values(0):
                close_prices = data['Adj Close']
            else:
                close_prices = data.iloc[:, :len(tickers)]
        else:
            if 'Close' in data.columns:
                close_prices = data[['Close']]
                close_prices.columns = tickers
            elif 'Adj Close' in data.columns:
                close_prices = data[['Adj Close']]
                close_prices.columns = tickers
            else:
                close_prices = data

        if isinstance(close_prices, pd.Series):
            close_prices = close_prices.to_frame()
            close_prices.columns = tickers

        close_prices = close_prices.dropna(how='all')

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

        if data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            if 'Volume' in data.columns.get_level_values(0):
                volume = data['Volume']
            else:
                return pd.DataFrame()
        else:
            if 'Volume' in data.columns:
                volume = data[['Volume']]
                volume.columns = tickers
            else:
                return pd.DataFrame()

        if isinstance(volume, pd.Series):
            volume = volume.to_frame()
            volume.columns = tickers

        volume = volume.dropna(how='all')

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
