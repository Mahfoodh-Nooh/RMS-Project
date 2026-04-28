import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class RiskEngine:
    
    def __init__(self, returns: pd.DataFrame, confidence_levels: List[float] = [0.95, 0.99]):
        
        self.returns = returns
        self.confidence_levels = confidence_levels
        self.risk_metrics = {}
    
    def calculate_returns_stats(self) -> Dict[str, pd.Series]:
        
        stats_dict = {
            'mean_return': self.returns.mean(),
            'std_dev': self.returns.std(),
            'annualized_return': self.returns.mean() * 252,
            'annualized_volatility': self.returns.std() * np.sqrt(252),
            'skewness': self.returns.skew(),
            'kurtosis': self.returns.kurtosis()
        }
        
        self.risk_metrics['returns_stats'] = stats_dict
        return stats_dict
    
    def calculate_correlation_matrix(self) -> pd.DataFrame:
        
        correlation_matrix = self.returns.corr()
        self.risk_metrics['correlation_matrix'] = correlation_matrix
        return correlation_matrix
    
    def calculate_covariance_matrix(self) -> pd.DataFrame:
        
        covariance_matrix = self.returns.cov()
        self.risk_metrics['covariance_matrix'] = covariance_matrix
        return covariance_matrix
    
    def calculate_ewma_volatility(self, span: int = 60) -> pd.DataFrame:
        
        ewma_volatility = self.returns.ewm(span=span).std() * np.sqrt(252)
        self.risk_metrics['ewma_volatility'] = ewma_volatility
        return ewma_volatility
    
    def calculate_rolling_volatility(self, window: int = 30) -> pd.DataFrame:
        
        rolling_volatility = self.returns.rolling(window=window).std() * np.sqrt(252)
        self.risk_metrics['rolling_volatility'] = rolling_volatility
        return rolling_volatility
    
    def calculate_parametric_var(
        self,
        portfolio_value: float = 1000000,
        weights: Optional[np.ndarray] = None
    ) -> Dict[float, float]:
        
        if weights is None:
            weights = np.array([1.0 / len(self.returns.columns)] * len(self.returns.columns))
        
        portfolio_returns = (self.returns * weights).sum(axis=1)
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        var_dict = {}
        for conf_level in self.confidence_levels:
            z_score = stats.norm.ppf(1 - conf_level)
            var = portfolio_value * (mean_return + z_score * std_return)
            var_dict[conf_level] = abs(var)
        
        self.risk_metrics['parametric_var'] = var_dict
        return var_dict
    
    def calculate_historical_var(
        self,
        portfolio_value: float = 1000000,
        weights: Optional[np.ndarray] = None
    ) -> Dict[float, float]:
        
        if weights is None:
            weights = np.array([1.0 / len(self.returns.columns)] * len(self.returns.columns))
        
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        var_dict = {}
        for conf_level in self.confidence_levels:
            var = np.percentile(portfolio_returns, (1 - conf_level) * 100)
            var_dict[conf_level] = abs(var * portfolio_value)
        
        self.risk_metrics['historical_var'] = var_dict
        return var_dict
    
    def calculate_cvar(
        self,
        portfolio_value: float = 1000000,
        weights: Optional[np.ndarray] = None
    ) -> Dict[float, float]:
        
        if weights is None:
            weights = np.array([1.0 / len(self.returns.columns)] * len(self.returns.columns))
        
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        cvar_dict = {}
        for conf_level in self.confidence_levels:
            var_threshold = np.percentile(portfolio_returns, (1 - conf_level) * 100)
            cvar = portfolio_returns[portfolio_returns <= var_threshold].mean()
            cvar_dict[conf_level] = abs(cvar * portfolio_value)
        
        self.risk_metrics['cvar'] = cvar_dict
        return cvar_dict
    
    def calculate_maximum_drawdown(
        self,
        prices: pd.DataFrame
    ) -> Dict[str, float]:
        
        cumulative_returns = (1 + self.returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_drawdown = drawdown.min()
        
        if isinstance(max_drawdown, pd.Series):
            max_dd_dict = max_drawdown.to_dict()
            portfolio_max_dd = max_drawdown.mean()
        else:
            max_dd_dict = {'portfolio': max_drawdown}
            portfolio_max_dd = max_drawdown
        
        result = {
            'by_asset': max_dd_dict,
            'portfolio': portfolio_max_dd,
            'drawdown_series': drawdown
        }
        
        self.risk_metrics['maximum_drawdown'] = result
        return result
    
    def monte_carlo_simulation(
        self,
        portfolio_value: float = 1000000,
        weights: Optional[np.ndarray] = None,
        num_simulations: int = 10000,
        time_horizon: int = 252
    ) -> Dict:
        
        if weights is None:
            weights = np.array([1.0 / len(self.returns.columns)] * len(self.returns.columns))
        
        mean_returns = self.returns.mean().values
        cov_matrix = self.returns.cov().values
        
        simulated_returns = np.random.multivariate_normal(
            mean_returns,
            cov_matrix,
            (num_simulations, time_horizon)
        )
        
        portfolio_simulated_returns = np.tensordot(simulated_returns, weights, axes=([2], [0]))
        
        cumulative_returns = np.cumprod(1 + portfolio_simulated_returns, axis=1)
        final_portfolio_values = portfolio_value * cumulative_returns[:, -1]
        
        mc_var_dict = {}
        for conf_level in self.confidence_levels:
            mc_var = portfolio_value - np.percentile(final_portfolio_values, (1 - conf_level) * 100)
            mc_var_dict[conf_level] = mc_var
        
        result = {
            'simulated_paths': cumulative_returns,
            'final_values': final_portfolio_values,
            'mc_var': mc_var_dict,
            'mean_final_value': final_portfolio_values.mean(),
            'std_final_value': final_portfolio_values.std(),
            'percentiles': {
                '5th': np.percentile(final_portfolio_values, 5),
                '50th': np.percentile(final_portfolio_values, 50),
                '95th': np.percentile(final_portfolio_values, 95)
            }
        }
        
        self.risk_metrics['monte_carlo'] = result
        return result
    
    def stress_test(
        self,
        portfolio_value: float = 1000000,
        weights: Optional[np.ndarray] = None,
        shock_scenarios: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        
        if weights is None:
            weights = np.array([1.0 / len(self.returns.columns)] * len(self.returns.columns))
        
        if shock_scenarios is None:
            shock_scenarios = {
                'mild_shock': -0.05,
                'moderate_shock': -0.10,
                'severe_shock': -0.20,
                'extreme_shock': -0.30
            }
        
        stress_results = {}
        for scenario_name, shock_value in shock_scenarios.items():
            shocked_value = portfolio_value * (1 + shock_value)
            loss = portfolio_value - shocked_value
            stress_results[scenario_name] = {
                'shocked_value': shocked_value,
                'loss': loss,
                'loss_percentage': shock_value * 100
            }
        
        self.risk_metrics['stress_test'] = stress_results
        return stress_results
    
    def calculate_liquidity_risk(
        self,
        volume_data: pd.DataFrame,
        prices: pd.DataFrame,
        portfolio_shares: Optional[Dict[str, float]] = None
    ) -> Dict:
        
        avg_daily_volume = volume_data.mean()
        avg_daily_dollar_volume = (volume_data * prices).mean()
        
        liquidity_metrics = {
            'avg_daily_volume': avg_daily_volume.to_dict(),
            'avg_daily_dollar_volume': avg_daily_dollar_volume.to_dict()
        }
        
        if portfolio_shares is not None:
            days_to_liquidate = {}
            for ticker, shares in portfolio_shares.items():
                if ticker in avg_daily_volume.index:
                    daily_vol = avg_daily_volume[ticker]
                    days_needed = shares / (daily_vol * 0.1)
                    days_to_liquidate[ticker] = days_needed
            
            liquidity_metrics['days_to_liquidate'] = days_to_liquidate
            liquidity_metrics['liquidity_score'] = self._calculate_liquidity_score(days_to_liquidate)
        
        self.risk_metrics['liquidity_risk'] = liquidity_metrics
        return liquidity_metrics
    
    def _calculate_liquidity_score(self, days_to_liquidate: Dict[str, float]) -> str:
        
        avg_days = np.mean(list(days_to_liquidate.values()))
        
        if avg_days < 5:
            return "HIGH_LIQUIDITY"
        elif avg_days < 10:
            return "MEDIUM_LIQUIDITY"
        else:
            return "LOW_LIQUIDITY"
    
    def calculate_sharpe_ratio(
        self,
        weights: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02
    ) -> float:
        
        if weights is None:
            weights = np.array([1.0 / len(self.returns.columns)] * len(self.returns.columns))
        
        portfolio_returns = (self.returns * weights).sum(axis=1)
        excess_returns = portfolio_returns.mean() * 252 - risk_free_rate
        portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        
        sharpe_ratio = excess_returns / portfolio_volatility
        self.risk_metrics['sharpe_ratio'] = sharpe_ratio
        return sharpe_ratio
    
    def calculate_sortino_ratio(
        self,
        weights: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.02
    ) -> float:
        
        if weights is None:
            weights = np.array([1.0 / len(self.returns.columns)] * len(self.returns.columns))
        
        portfolio_returns = (self.returns * weights).sum(axis=1)
        excess_returns = portfolio_returns.mean() * 252 - risk_free_rate
        
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        sortino_ratio = excess_returns / downside_deviation if downside_deviation != 0 else 0
        self.risk_metrics['sortino_ratio'] = sortino_ratio
        return sortino_ratio
    
    def calculate_beta(
        self,
        market_returns: pd.Series,
        weights: Optional[np.ndarray] = None
    ) -> float:
        
        if weights is None:
            weights = np.array([1.0 / len(self.returns.columns)] * len(self.returns.columns))
        
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        aligned_returns = pd.DataFrame({
            'portfolio': portfolio_returns,
            'market': market_returns
        }).dropna()
        
        covariance = aligned_returns.cov().loc['portfolio', 'market']
        market_variance = aligned_returns['market'].var()
        
        beta = covariance / market_variance
        self.risk_metrics['beta'] = beta
        return beta
    
    def generate_risk_report(self) -> Dict:
        
        return self.risk_metrics
    
    def get_risk_summary(
        self,
        portfolio_value: float = 1000000,
        weights: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        
        self.calculate_returns_stats()
        self.calculate_correlation_matrix()
        var_param = self.calculate_parametric_var(portfolio_value, weights)
        var_hist = self.calculate_historical_var(portfolio_value, weights)
        cvar = self.calculate_cvar(portfolio_value, weights)
        sharpe = self.calculate_sharpe_ratio(weights)
        sortino = self.calculate_sortino_ratio(weights)
        
        summary_data = {
            'Metric': [],
            'Value': []
        }
        
        summary_data['Metric'].append('Sharpe Ratio')
        summary_data['Value'].append(f"{sharpe:.4f}")
        
        summary_data['Metric'].append('Sortino Ratio')
        summary_data['Value'].append(f"{sortino:.4f}")
        
        for conf in self.confidence_levels:
            summary_data['Metric'].append(f'Parametric VaR ({conf*100}%)')
            summary_data['Value'].append(f"${var_param[conf]:,.2f}")
            
            summary_data['Metric'].append(f'Historical VaR ({conf*100}%)')
            summary_data['Value'].append(f"${var_hist[conf]:,.2f}")
            
            summary_data['Metric'].append(f'CVaR ({conf*100}%)')
            summary_data['Value'].append(f"${cvar[conf]:,.2f}")
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
