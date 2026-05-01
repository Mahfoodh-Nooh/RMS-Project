import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple


class SymbolUniverse:

    def __init__(self, data_dir: str = '../data'):
        self.data_dir = data_dir
        self._tasi: Optional[pd.DataFrame] = None
        self._global: Optional[pd.DataFrame] = None

    def _load(self, filename: str) -> pd.DataFrame:
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
            return df
        return pd.DataFrame(columns=['Symbol', 'Company', 'Sector', 'Market'])

    @property
    def tasi(self) -> pd.DataFrame:
        if self._tasi is None:
            self._tasi = self._load('tasi_symbols.csv')
        return self._tasi

    @property
    def global_stocks(self) -> pd.DataFrame:
        if self._global is None:
            self._global = self._load('global_symbols.csv')
        return self._global

    def get_all(self) -> pd.DataFrame:
        return pd.concat([self.tasi, self.global_stocks], ignore_index=True)

    def get_display_options(self, market: str = 'all') -> Dict[str, str]:
        if market == 'tasi':
            df = self.tasi
        elif market == 'global':
            df = self.global_stocks
        else:
            df = self.get_all()

        if df.empty:
            return {}

        options = {}
        for _, row in df.iterrows():
            label = f"{row['Symbol']} - {row.get('Company', '')}"
            options[label] = row['Symbol']
        return options

    def is_valid(self, symbol: str, market: str = 'all') -> bool:
        symbol = symbol.strip().upper()
        df = self.tasi if market == 'tasi' else (
            self.global_stocks if market == 'global' else self.get_all()
        )
        return symbol in df['Symbol'].values

    def get_company_name(self, symbol: str) -> str:
        df = self.get_all()
        row = df[df['Symbol'] == symbol.upper()]
        if not row.empty:
            return row.iloc[0].get('Company', symbol)
        return symbol

    def validate_symbols(self, symbols: List[str], market: str = 'all') -> Tuple[List[str], List[str]]:
        valid, invalid = [], []
        for s in symbols:
            (valid if self.is_valid(s, market) else invalid).append(s)
        return valid, invalid


class ValidationEngine:

    def __init__(self, universe: Optional[SymbolUniverse] = None):
        self.universe = universe

    def validate_positions_df(self, df: pd.DataFrame) -> Tuple[bool, List[str], pd.DataFrame]:
        errors = []

        required_cols = ['Symbol', 'Quantity']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            errors.append(f"Missing columns: {missing}")
            return False, errors, df

        df = df.copy()
        df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
        df = df[df['Symbol'].notna() & (df['Symbol'] != '') & (df['Symbol'] != 'NAN')]

        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        bad_qty = df[df['Quantity'].isna() | (df['Quantity'] <= 0)]['Symbol'].tolist()
        if bad_qty:
            errors.append(f"Invalid or zero quantity for: {bad_qty}")
            df = df[df['Quantity'] > 0].copy()

        if df.empty:
            errors.append("No valid positions remain after validation.")
            return False, errors, df

        dupes = df[df.duplicated(subset='Symbol')]['Symbol'].tolist()
        if dupes:
            errors.append(f"Duplicate symbols removed: {dupes}")
            df = df.drop_duplicates(subset='Symbol')

        return True, errors, df.reset_index(drop=True)

    def validate_weights(self, weights: np.ndarray, tolerance: float = 1e-4) -> Tuple[np.ndarray, str]:
        weights = np.array(weights, dtype=float)
        weights = np.where(weights < 0, 0, weights)
        total = weights.sum()
        if total == 0:
            n = len(weights)
            return np.ones(n) / n, "All weights were zero — equal weights applied."
        if abs(total - 1.0) > tolerance:
            weights = weights / total
            return weights, f"Weights normalized (sum was {total:.4f})."
        return weights, "OK"

    def check_concentration(self, weights: np.ndarray, symbols: List[str], threshold: float = 0.40) -> List[Dict]:
        alerts = []
        for i, (w, s) in enumerate(zip(weights, symbols)):
            if w > threshold:
                alerts.append({
                    'type': 'CONCENTRATION',
                    'severity': 'HIGH',
                    'symbol': s,
                    'weight': w,
                    'message': f"High concentration: {s} represents {w*100:.1f}% of portfolio (>{threshold*100:.0f}% threshold)"
                })
        return alerts
