import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import DataLoader
from risk_engine import RiskEngine
from portfolio import Portfolio
from ml_model import RiskPredictor
import numpy as np


def main():
    print("=" * 60)
    print("Risk Management System (RMS) - Example Usage")
    print("=" * 60)
    print()
    
    tickers = ['2222.SR', '1120.SR', '2010.SR']
    print(f"Analyzing Saudi stocks: {', '.join(tickers)}")
    print()
    
    print("Step 1: Loading market data...")
    loader = DataLoader(data_dir='../data')
    prices = loader.get_close_prices(tickers, period='1y')
    returns = loader.get_returns(tickers, period='1y')
    volume = loader.get_volume_data(tickers, period='1y')
    print(f"✓ Loaded {len(prices)} days of data")
    print()
    
    print("Step 2: Creating portfolio...")
    portfolio_value = 1000000
    weights = np.array([0.4, 0.3, 0.3])
    
    portfolio = Portfolio(
        name="Saudi Portfolio Example",
        initial_value=portfolio_value,
        tickers=tickers,
        weights=weights
    )
    
    latest_prices = prices.iloc[-1]
    portfolio.update_prices(latest_prices)
    print(f"✓ Portfolio created with value: ${portfolio.current_value:,.2f}")
    print()
    
    print("Step 3: Calculating risk metrics...")
    risk_engine = RiskEngine(returns, confidence_levels=[0.95, 0.99])
    
    returns_stats = risk_engine.calculate_returns_stats()
    print(f"  Annualized Return: {returns_stats['annualized_return'].mean():.2%}")
    print(f"  Annualized Volatility: {returns_stats['annualized_volatility'].mean():.2%}")
    
    var_param = risk_engine.calculate_parametric_var(portfolio_value, weights)
    var_hist = risk_engine.calculate_historical_var(portfolio_value, weights)
    cvar = risk_engine.calculate_cvar(portfolio_value, weights)
    
    print(f"\n  VaR (95% confidence):")
    print(f"    Parametric: ${var_param[0.95]:,.2f}")
    print(f"    Historical: ${var_hist[0.95]:,.2f}")
    print(f"    CVaR: ${cvar[0.95]:,.2f}")
    
    sharpe = risk_engine.calculate_sharpe_ratio(weights)
    sortino = risk_engine.calculate_sortino_ratio(weights)
    print(f"\n  Sharpe Ratio: {sharpe:.4f}")
    print(f"  Sortino Ratio: {sortino:.4f}")
    print()
    
    print("Step 4: Running Monte Carlo simulation...")
    mc_results = risk_engine.monte_carlo_simulation(
        portfolio_value=portfolio_value,
        weights=weights,
        num_simulations=5000,
        time_horizon=252
    )
    print(f"  Mean Final Value: ${mc_results['mean_final_value']:,.2f}")
    print(f"  5th Percentile: ${mc_results['percentiles']['5th']:,.2f}")
    print(f"  95th Percentile: ${mc_results['percentiles']['95th']:,.2f}")
    print()
    
    print("Step 5: Stress testing...")
    stress_results = risk_engine.stress_test(portfolio_value, weights)
    for scenario, result in stress_results.items():
        print(f"  {scenario.replace('_', ' ').title()}: Loss of ${result['loss']:,.2f} ({result['loss_percentage']:.1f}%)")
    print()
    
    print("Step 6: Training ML risk predictor...")
    predictor = RiskPredictor(model_type='classification')
    
    features = predictor.prepare_features(returns, prices, volume)
    target = predictor.create_target_classification(returns, weights)
    
    train_results = predictor.train_model(
        features,
        target,
        test_size=0.2,
        use_xgboost=False
    )
    
    print(f"  Training Accuracy: {train_results['train_accuracy']:.4f}")
    print(f"  Test Accuracy: {train_results['test_accuracy']:.4f}")
    
    latest_features = features.iloc[-1:]
    risk_level = predictor.predict_risk_level(latest_features)
    print(f"  Current Predicted Risk Level: {risk_level[0]}")
    print()
    
    print("Step 7: Generating comprehensive report...")
    risk_summary = risk_engine.get_risk_summary(portfolio_value, weights)
    print("\n" + str(risk_summary))
    print()
    
    print("=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Run 'streamlit run dashboard/app.py' for interactive dashboard")
    print("2. Explore Jupyter notebooks in 'notebooks/' directory")
    print("3. Customize risk metrics in 'src/risk_engine.py'")
    print()


if __name__ == "__main__":
    main()
