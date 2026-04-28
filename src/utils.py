import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json


def format_currency(value: float, currency: str = 'USD') -> str:
    
    if currency == 'SAR':
        return f"SAR {value:,.2f}"
    else:
        return f"${value:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    
    return f"{value * 100:.{decimals}f}%"


def calculate_returns(prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
    
    if method == 'simple':
        returns = prices.pct_change()
    elif method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        raise ValueError("method must be 'simple' or 'log'")
    
    return returns.dropna()


def annualize_returns(returns: pd.Series, periods_per_year: int = 252) -> float:
    
    return returns.mean() * periods_per_year


def annualize_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    
    return returns.std() * np.sqrt(periods_per_year)


def calculate_cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    
    return (1 + returns).cumprod() - 1


def calculate_rolling_metrics(
    returns: pd.DataFrame,
    window: int = 30,
    metrics: List[str] = ['mean', 'std', 'sharpe']
) -> Dict[str, pd.DataFrame]:
    
    results = {}
    
    if 'mean' in metrics:
        results['rolling_mean'] = returns.rolling(window).mean() * 252
    
    if 'std' in metrics:
        results['rolling_std'] = returns.rolling(window).std() * np.sqrt(252)
    
    if 'sharpe' in metrics:
        rolling_mean = returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        results['rolling_sharpe'] = rolling_mean / rolling_std
    
    return results


def generate_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = 'Correlation Matrix',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def detect_high_correlation(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.7
) -> List[Tuple[str, str, float]]:
    
    high_corr = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) >= threshold:
                high_corr.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    corr_value
                ))
    
    return high_corr


def generate_alert(
    alert_type: str,
    message: str,
    severity: str = 'INFO',
    timestamp: Optional[datetime] = None
) -> Dict:
    
    if timestamp is None:
        timestamp = datetime.now()
    
    alert = {
        'timestamp': timestamp.isoformat(),
        'type': alert_type,
        'severity': severity,
        'message': message
    }
    
    return alert


def check_var_threshold(
    var_value: float,
    threshold: float,
    confidence_level: float = 0.95
) -> Optional[Dict]:
    
    if var_value > threshold:
        return generate_alert(
            alert_type='VAR_THRESHOLD_EXCEEDED',
            message=f"VaR at {confidence_level*100}% confidence (${var_value:,.2f}) exceeds threshold (${threshold:,.2f})",
            severity='HIGH'
        )
    return None


def check_liquidity_risk(
    liquidity_score: str,
    days_to_liquidate: Dict[str, float]
) -> Optional[Dict]:
    
    if liquidity_score == 'LOW_LIQUIDITY':
        max_days = max(days_to_liquidate.values()) if days_to_liquidate else 0
        ticker_max = max(days_to_liquidate, key=days_to_liquidate.get) if days_to_liquidate else 'Unknown'
        
        return generate_alert(
            alert_type='LIQUIDITY_RISK',
            message=f"Low liquidity detected. {ticker_max} requires {max_days:.1f} days to liquidate.",
            severity='MEDIUM'
        )
    return None


def check_ml_risk_signal(risk_level: str) -> Optional[Dict]:
    
    if risk_level == 'HIGH':
        return generate_alert(
            alert_type='ML_RISK_SIGNAL',
            message="ML model predicts HIGH risk level for the portfolio.",
            severity='HIGH'
        )
    return None


def calculate_portfolio_metrics_summary(
    returns: pd.DataFrame,
    weights: np.ndarray,
    risk_free_rate: float = 0.02
) -> Dict:
    
    portfolio_returns = (returns * weights).sum(axis=1)
    
    total_return = (1 + portfolio_returns).prod() - 1
    ann_return = portfolio_returns.mean() * 252
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol != 0 else 0
    
    cumulative = (1 + portfolio_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    return {
        'total_return': total_return,
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }


def validate_weights(weights: np.ndarray, tolerance: float = 1e-6) -> bool:
    
    return abs(weights.sum() - 1.0) <= tolerance and all(w >= 0 for w in weights)


def rebalance_weights(current_weights: np.ndarray) -> np.ndarray:
    
    if current_weights.sum() == 0:
        raise ValueError("Sum of weights cannot be zero")
    
    return current_weights / current_weights.sum()


def get_tadawul_top_stocks() -> List[str]:
    
    return [
        '2222.SR',
        '1120.SR',
        '2010.SR',
        '2030.SR',
        '1210.SR',
        '1111.SR',
        '2382.SR',
        '1180.SR',
        '2220.SR',
        '4030.SR'
    ]


def export_to_json(data: Dict, filepath: str):
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, default=str)
    print(f"Data exported to {filepath}")


def import_from_json(filepath: str) -> Dict:
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def calculate_risk_adjusted_return(
    returns: float,
    volatility: float,
    risk_free_rate: float = 0.02
) -> float:
    
    return (returns - risk_free_rate) / volatility if volatility != 0 else 0


def date_range_validator(start_date: str, end_date: str) -> bool:
    
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        return start < end
    except ValueError:
        return False


def get_market_trading_days(start_date: str, end_date: str) -> int:
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    business_days = pd.bdate_range(start, end)
    return len(business_days)


def calculate_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    
    active_returns = portfolio_returns - benchmark_returns
    tracking_error = active_returns.std() * np.sqrt(252)
    
    if tracking_error == 0:
        return 0
    
    return (active_returns.mean() * 252) / tracking_error


def calculate_calmar_ratio(
    annualized_return: float,
    max_drawdown: float
) -> float:
    
    if max_drawdown == 0:
        return 0
    
    return annualized_return / abs(max_drawdown)


def format_risk_report(risk_metrics: Dict) -> str:
    
    report = []
    report.append("=" * 60)
    report.append("RISK MANAGEMENT REPORT")
    report.append("=" * 60)
    report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    if 'sharpe_ratio' in risk_metrics:
        report.append(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.4f}")
    
    if 'sortino_ratio' in risk_metrics:
        report.append(f"Sortino Ratio: {risk_metrics['sortino_ratio']:.4f}")
    
    if 'parametric_var' in risk_metrics:
        report.append("\nParametric VaR:")
        for conf, value in risk_metrics['parametric_var'].items():
            report.append(f"  {conf*100}% confidence: ${value:,.2f}")
    
    if 'historical_var' in risk_metrics:
        report.append("\nHistorical VaR:")
        for conf, value in risk_metrics['historical_var'].items():
            report.append(f"  {conf*100}% confidence: ${value:,.2f}")
    
    if 'cvar' in risk_metrics:
        report.append("\nConditional VaR (CVaR):")
        for conf, value in risk_metrics['cvar'].items():
            report.append(f"  {conf*100}% confidence: ${value:,.2f}")
    
    report.append("=" * 60)
    
    return "\n".join(report)


def color_code_risk_level(risk_level: str) -> str:
    
    colors = {
        'LOW': 'green',
        'MEDIUM': 'orange',
        'HIGH': 'red'
    }
    return colors.get(risk_level, 'gray')
