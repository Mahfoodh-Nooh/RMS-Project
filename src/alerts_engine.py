import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime


class AlertsEngine:

    def __init__(
        self,
        var_threshold_pct: float = 0.05,
        correlation_threshold: float = 0.80,
        concentration_threshold: float = 0.40,
        volatility_threshold: float = 0.30
    ):
        self.var_threshold_pct = var_threshold_pct
        self.correlation_threshold = correlation_threshold
        self.concentration_threshold = concentration_threshold
        self.volatility_threshold = volatility_threshold
        self.alerts: List[Dict] = []

    def _add(self, alert_type: str, severity: str, message: str, detail: str = ""):
        self.alerts.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'type': alert_type,
            'severity': severity,
            'message': message,
            'detail': detail
        })

    def check_var(self, var_value: float, portfolio_value: float, confidence: float = 0.95):
        threshold = portfolio_value * self.var_threshold_pct
        if var_value > threshold:
            self._add(
                'VAR_EXCEEDED', 'HIGH',
                f"VaR ({confidence*100:.0f}%) exceeds {self.var_threshold_pct*100:.0f}% threshold",
                f"VaR = ${var_value:,.0f} | Threshold = ${threshold:,.0f}"
            )

    def check_correlation(self, corr_matrix: pd.DataFrame):
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if abs(val) >= self.correlation_threshold:
                    self._add(
                        'HIGH_CORRELATION', 'MEDIUM',
                        f"High correlation between {cols[i]} and {cols[j]}",
                        f"Correlation = {val:.3f} (threshold ≥ {self.correlation_threshold})"
                    )

    def check_concentration(self, weights: np.ndarray, symbols: List[str]):
        for w, s in zip(weights, symbols):
            if w > self.concentration_threshold:
                self._add(
                    'CONCENTRATION_RISK', 'HIGH',
                    f"High concentration in {s}: {w*100:.1f}%",
                    f"Single asset exceeds {self.concentration_threshold*100:.0f}% of portfolio"
                )

    def check_volatility(self, ann_volatility: float, symbol: str = "Portfolio"):
        if ann_volatility > self.volatility_threshold:
            self._add(
                'HIGH_VOLATILITY', 'MEDIUM',
                f"High annualized volatility detected: {ann_volatility*100:.1f}%",
                f"{symbol} volatility exceeds {self.volatility_threshold*100:.0f}% threshold"
            )

    def check_drawdown(self, max_drawdown: float, threshold: float = -0.20):
        if max_drawdown < threshold:
            self._add(
                'LARGE_DRAWDOWN', 'HIGH',
                f"Large maximum drawdown: {max_drawdown*100:.1f}%",
                f"Drawdown exceeds {threshold*100:.0f}% threshold"
            )

    def check_ml_signal(self, risk_level: str):
        if risk_level == 'HIGH':
            self._add(
                'ML_HIGH_RISK', 'HIGH',
                "ML model predicts HIGH risk level",
                "Consider reducing exposure or hedging the portfolio"
            )

    def run_all_checks(
        self,
        var_value: float,
        portfolio_value: float,
        corr_matrix: pd.DataFrame,
        weights: np.ndarray,
        symbols: List[str],
        ann_volatility: float,
        max_drawdown: float,
        ml_risk_level: Optional[str] = None
    ) -> List[Dict]:
        self.alerts = []
        self.check_var(var_value, portfolio_value)
        self.check_correlation(corr_matrix)
        self.check_concentration(weights, symbols)
        self.check_volatility(ann_volatility)
        self.check_drawdown(max_drawdown)
        if ml_risk_level:
            self.check_ml_signal(ml_risk_level)
        return self.alerts

    def get_alerts_df(self) -> pd.DataFrame:
        if not self.alerts:
            return pd.DataFrame()
        return pd.DataFrame(self.alerts)

    def get_severity_count(self) -> Dict[str, int]:
        counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for a in self.alerts:
            sev = a.get('severity', 'LOW')
            counts[sev] = counts.get(sev, 0) + 1
        return counts


class RecommendationEngine:

    def __init__(self):
        self.recommendations: List[Dict] = []

    def _add(self, category: str, message: str, action: str, priority: str = 'MEDIUM'):
        self.recommendations.append({
            'category': category,
            'message': message,
            'action': action,
            'priority': priority
        })

    def analyze(
        self,
        weights: np.ndarray,
        symbols: List[str],
        corr_matrix: pd.DataFrame,
        ann_volatility: float,
        sharpe_ratio: float,
        max_drawdown: float,
        var_pct: float
    ):
        self.recommendations = []

        max_w = weights.max() if len(weights) > 0 else 0
        if max_w > 0.40:
            top = symbols[int(np.argmax(weights))]
            self._add(
                'Diversification',
                f"High concentration in {top} ({max_w*100:.1f}%)",
                f"Consider reducing {top} below 30% and redistributing to other assets",
                'HIGH'
            )

        if len(symbols) >= 2:
            off_diag = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            avg_corr = np.mean(np.abs(off_diag)) if len(off_diag) > 0 else 0
            if avg_corr > 0.70:
                self._add(
                    'Correlation',
                    f"Portfolio is highly correlated (avg = {avg_corr:.2f})",
                    "Add uncorrelated assets such as bonds, commodities, or assets from different sectors",
                    'HIGH'
                )

        if ann_volatility > 0.25:
            self._add(
                'Volatility',
                f"Portfolio volatility is elevated ({ann_volatility*100:.1f}%)",
                "Consider adding defensive or low-volatility assets to reduce overall portfolio risk",
                'MEDIUM'
            )

        if sharpe_ratio < 0.5:
            self._add(
                'Risk-Return',
                f"Low Sharpe ratio ({sharpe_ratio:.2f}) indicates poor risk-adjusted returns",
                "Review asset selection and consider replacing underperforming assets",
                'MEDIUM'
            )

        if max_drawdown < -0.20:
            self._add(
                'Drawdown',
                f"Large historical drawdown detected ({max_drawdown*100:.1f}%)",
                "Consider implementing stop-loss strategies or adding hedging positions",
                'HIGH'
            )

        if len(symbols) < 5:
            self._add(
                'Diversification',
                f"Portfolio has only {len(symbols)} asset(s)",
                "Increase the number of holdings to at least 8-10 for better diversification",
                'MEDIUM'
            )

        if var_pct > 0.05:
            self._add(
                'VaR',
                f"Daily VaR is high ({var_pct*100:.1f}% of portfolio)",
                "Reduce position sizes or shift to lower-volatility instruments",
                'HIGH'
            )

        if not self.recommendations:
            self._add(
                'General',
                "Portfolio risk profile appears acceptable",
                "Continue monitoring and rebalance periodically (quarterly recommended)",
                'LOW'
            )

        return self.recommendations

    def get_df(self) -> pd.DataFrame:
        if not self.recommendations:
            return pd.DataFrame()
        return pd.DataFrame(self.recommendations)
