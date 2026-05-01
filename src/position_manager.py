import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import os


class PositionManager:

    EXCEL_TEMPLATE_COLS = ['Symbol', 'Quantity']

    def __init__(self, prices_cache: Optional[Dict[str, float]] = None):
        self.positions: pd.DataFrame = pd.DataFrame()
        self.prices_cache = prices_cache or {}

    def from_excel(self, file) -> Tuple[bool, str, pd.DataFrame]:
        try:
            df = pd.read_excel(file)
        except Exception as e:
            return False, f"Cannot read file: {str(e)}", pd.DataFrame()

        df.columns = [c.strip().title() for c in df.columns]

        missing = [c for c in self.EXCEL_TEMPLATE_COLS if c not in df.columns]
        if missing:
            return False, f"Missing columns: {missing}. Required: {self.EXCEL_TEMPLATE_COLS}", pd.DataFrame()

        df = df[self.EXCEL_TEMPLATE_COLS].copy()
        df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
        df = df[df['Symbol'].notna() & (df['Symbol'] != '') & (df['Symbol'] != 'NAN')]
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        invalid_qty = df[df['Quantity'].isna() | (df['Quantity'] <= 0)]
        if not invalid_qty.empty:
            bad = invalid_qty['Symbol'].tolist()
            df = df[df['Quantity'] > 0].copy()
            if df.empty:
                return False, f"No valid quantities found. Invalid rows: {bad}", pd.DataFrame()

        df = df.drop_duplicates(subset='Symbol').reset_index(drop=True)
        self.positions = df
        return True, "OK", df

    def from_manual(self, symbols: List[str], quantities: Optional[List[float]] = None) -> Tuple[bool, str, pd.DataFrame]:
        if not symbols:
            return False, "No symbols provided.", pd.DataFrame()

        symbols = [s.strip().upper() for s in symbols if s.strip()]

        if quantities is None or len(quantities) != len(symbols):
            quantities = [1.0] * len(symbols)

        df = pd.DataFrame({'Symbol': symbols, 'Quantity': quantities})
        df = df[df['Quantity'] > 0].reset_index(drop=True)
        self.positions = df
        return True, "OK", df

    def fetch_prices(self, period: str = '1d') -> Tuple[pd.DataFrame, List[str]]:
        if self.positions.empty:
            return pd.DataFrame(), []

        symbols = self.positions['Symbol'].tolist()
        failed = []
        price_map = {}

        for sym in symbols:
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(period='5d')
                if not hist.empty:
                    price_map[sym] = float(hist['Close'].iloc[-1])
                else:
                    failed.append(sym)
                    price_map[sym] = np.nan
            except Exception:
                failed.append(sym)
                price_map[sym] = np.nan

        self.positions['Price'] = self.positions['Symbol'].map(price_map)
        self.positions['Value'] = self.positions['Price'] * self.positions['Quantity']

        total = self.positions['Value'].sum()
        self.positions['Weight'] = self.positions['Value'] / total if total > 0 else 1.0 / len(symbols)

        return self.positions, failed

    def get_tickers(self) -> List[str]:
        return self.positions['Symbol'].tolist() if not self.positions.empty else []

    def get_weights(self) -> np.ndarray:
        if self.positions.empty or 'Weight' not in self.positions.columns:
            n = len(self.positions)
            return np.array([1.0 / n] * n) if n > 0 else np.array([])
        w = self.positions['Weight'].fillna(0).values
        s = w.sum()
        return w / s if s > 0 else np.ones(len(w)) / len(w)

    def get_portfolio_value(self) -> float:
        if 'Value' in self.positions.columns:
            return float(self.positions['Value'].sum())
        return 0.0

    def get_summary_df(self) -> pd.DataFrame:
        if self.positions.empty:
            return pd.DataFrame()
        df = self.positions.copy()
        if 'Price' in df.columns:
            df['Price'] = df['Price'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
        if 'Value' in df.columns:
            df['Value'] = df['Value'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
        if 'Weight' in df.columns:
            df['Weight'] = df['Weight'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
        return df

    @staticmethod
    def generate_excel_template() -> bytes:
        df = pd.DataFrame({
            'Symbol': ['2222.SR', '1120.SR', '2010.SR', 'AAPL', 'MSFT'],
            'Quantity': [1000, 500, 300, 200, 150]
        })
        from io import BytesIO
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Portfolio')
        return buf.getvalue()
