import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class Portfolio:
    
    def __init__(
        self,
        name: str,
        initial_value: float,
        tickers: List[str],
        weights: Optional[np.ndarray] = None
    ):
        
        self.name = name
        self.initial_value = initial_value
        self.current_value = initial_value
        self.tickers = tickers
        
        if weights is None:
            self.weights = np.array([1.0 / len(tickers)] * len(tickers))
        else:
            self.weights = np.array(weights)
            self.weights = self.weights / self.weights.sum()
        
        self.holdings = {}
        self.allocation = {}
        self._initialize_allocation()
        
        self.history = []
        self.transactions = []
    
    def _initialize_allocation(self):
        
        for i, ticker in enumerate(self.tickers):
            allocation_value = self.initial_value * self.weights[i]
            self.allocation[ticker] = {
                'weight': self.weights[i],
                'value': allocation_value,
                'percentage': self.weights[i] * 100
            }
    
    def update_prices(self, current_prices: pd.Series):
        
        total_value = 0
        for ticker in self.tickers:
            if ticker in current_prices.index:
                idx = self.tickers.index(ticker)
                shares = self.allocation[ticker]['value'] / current_prices[ticker]
                current_value = shares * current_prices[ticker]
                self.allocation[ticker]['current_value'] = current_value
                self.allocation[ticker]['shares'] = shares
                total_value += current_value
        
        self.current_value = total_value
        
        for ticker in self.tickers:
            if 'current_value' in self.allocation[ticker]:
                self.allocation[ticker]['current_weight'] = (
                    self.allocation[ticker]['current_value'] / total_value
                )
                self.allocation[ticker]['current_percentage'] = (
                    self.allocation[ticker]['current_weight'] * 100
                )
        
        self.history.append({
            'timestamp': datetime.now(),
            'portfolio_value': total_value,
            'allocation': self.allocation.copy()
        })
    
    def rebalance(
        self,
        new_weights: np.ndarray,
        current_prices: pd.Series,
        rebalance_cost: float = 0.001
    ) -> Dict:
        
        new_weights = np.array(new_weights)
        new_weights = new_weights / new_weights.sum()
        
        old_allocation = self.allocation.copy()
        total_cost = 0
        
        for i, ticker in enumerate(self.tickers):
            if ticker in current_prices.index:
                old_value = self.allocation[ticker].get('current_value', self.allocation[ticker]['value'])
                new_value = self.current_value * new_weights[i]
                trade_value = abs(new_value - old_value)
                total_cost += trade_value * rebalance_cost
                
                self.allocation[ticker]['weight'] = new_weights[i]
                self.allocation[ticker]['value'] = new_value
                self.allocation[ticker]['percentage'] = new_weights[i] * 100
        
        self.weights = new_weights
        self.current_value -= total_cost
        
        self.transactions.append({
            'timestamp': datetime.now(),
            'type': 'rebalance',
            'cost': total_cost,
            'old_weights': [old_allocation[t]['weight'] for t in self.tickers],
            'new_weights': new_weights.tolist()
        })
        
        return {
            'rebalance_cost': total_cost,
            'old_allocation': old_allocation,
            'new_allocation': self.allocation.copy()
        }
    
    def add_cash(self, amount: float):
        
        self.current_value += amount
        self.initial_value += amount
        
        for ticker in self.tickers:
            self.allocation[ticker]['value'] += amount * self.allocation[ticker]['weight']
        
        self.transactions.append({
            'timestamp': datetime.now(),
            'type': 'cash_deposit',
            'amount': amount
        })
    
    def withdraw_cash(self, amount: float) -> bool:
        
        if amount > self.current_value:
            return False
        
        self.current_value -= amount
        
        for ticker in self.tickers:
            self.allocation[ticker]['value'] -= amount * self.allocation[ticker]['weight']
        
        self.transactions.append({
            'timestamp': datetime.now(),
            'type': 'cash_withdrawal',
            'amount': amount
        })
        
        return True
    
    def get_portfolio_summary(self) -> pd.DataFrame:
        
        summary_data = []
        for ticker in self.tickers:
            alloc = self.allocation[ticker]
            summary_data.append({
                'Ticker': ticker,
                'Target Weight': f"{alloc['weight']*100:.2f}%",
                'Target Value': f"${alloc['value']:,.2f}",
                'Current Weight': f"{alloc.get('current_weight', alloc['weight'])*100:.2f}%",
                'Current Value': f"${alloc.get('current_value', alloc['value']):,.2f}",
                'Shares': f"{alloc.get('shares', 0):.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df
    
    def get_performance_metrics(self, returns: pd.DataFrame) -> Dict:
        
        portfolio_returns = (returns * self.weights).sum(axis=1)
        
        cumulative_return = (1 + portfolio_returns).prod() - 1
        annualized_return = portfolio_returns.mean() * 252
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        positive_returns = portfolio_returns[portfolio_returns > 0]
        negative_returns = portfolio_returns[portfolio_returns < 0]
        
        win_rate = len(positive_returns) / len(portfolio_returns) if len(portfolio_returns) > 0 else 0
        
        metrics = {
            'total_return': cumulative_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': annualized_return / annualized_volatility if annualized_volatility != 0 else 0,
            'win_rate': win_rate,
            'current_value': self.current_value,
            'initial_value': self.initial_value,
            'profit_loss': self.current_value - self.initial_value,
            'profit_loss_percentage': ((self.current_value - self.initial_value) / self.initial_value) * 100
        }
        
        return metrics
    
    def optimize_weights_min_variance(self, returns: pd.DataFrame) -> np.ndarray:
        
        cov_matrix = returns.cov().values
        n_assets = len(self.tickers)
        
        inv_cov = np.linalg.inv(cov_matrix)
        ones = np.ones(n_assets)
        
        optimal_weights = inv_cov @ ones / (ones @ inv_cov @ ones)
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        return optimal_weights
    
    def optimize_weights_max_sharpe(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> np.ndarray:
        
        mean_returns = returns.mean().values * 252
        cov_matrix = returns.cov().values * 252
        
        n_assets = len(self.tickers)
        
        from scipy.optimize import minimize
        
        def negative_sharpe(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - risk_free_rate) / portfolio_std
            return -sharpe
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_guess = np.array([1.0 / n_assets] * n_assets)
        
        result = minimize(
            negative_sharpe,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def get_allocation_drift(self) -> pd.DataFrame:
        
        drift_data = []
        for ticker in self.tickers:
            target_weight = self.allocation[ticker]['weight']
            current_weight = self.allocation[ticker].get('current_weight', target_weight)
            drift = current_weight - target_weight
            
            drift_data.append({
                'Ticker': ticker,
                'Target Weight': f"{target_weight*100:.2f}%",
                'Current Weight': f"{current_weight*100:.2f}%",
                'Drift': f"{drift*100:.2f}%",
                'Needs Rebalance': 'Yes' if abs(drift) > 0.05 else 'No'
            })
        
        drift_df = pd.DataFrame(drift_data)
        return drift_df
    
    def get_transaction_history(self) -> pd.DataFrame:
        
        if not self.transactions:
            return pd.DataFrame()
        
        return pd.DataFrame(self.transactions)
    
    def export_portfolio(self) -> Dict:
        
        portfolio_data = {
            'name': self.name,
            'initial_value': self.initial_value,
            'current_value': self.current_value,
            'tickers': self.tickers,
            'weights': self.weights.tolist(),
            'allocation': self.allocation,
            'transactions': self.transactions,
            'history': self.history
        }
        
        return portfolio_data
    
    @classmethod
    def import_portfolio(cls, portfolio_data: Dict) -> 'Portfolio':
        
        portfolio = cls(
            name=portfolio_data['name'],
            initial_value=portfolio_data['initial_value'],
            tickers=portfolio_data['tickers'],
            weights=np.array(portfolio_data['weights'])
        )
        
        portfolio.current_value = portfolio_data['current_value']
        portfolio.allocation = portfolio_data['allocation']
        portfolio.transactions = portfolio_data.get('transactions', [])
        portfolio.history = portfolio_data.get('history', [])
        
        return portfolio
