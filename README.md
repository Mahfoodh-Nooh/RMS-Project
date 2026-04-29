# 📊 Risk Management System (RMS)

Professional Risk Management System for equity portfolio management with focus on Saudi Stock Market (Tadawul) and global compatibility.

## 🎯 Overview

A production-ready, data-driven Risk Management System that provides:
- **Preventive Risk Analysis**: Predictive insights using machine learning
- **Comprehensive Risk Metrics**: VaR, CVaR, Stress Testing, Monte Carlo Simulation
- **Interactive Dashboard**: Real-time risk monitoring and visualization
- **Saudi Market Ready**: Optimized for Tadawul stocks with global compatibility
- **Scalable Architecture**: Modular design ready for SaaS evolution

---

## 🏗️ System Architecture

```
RMS_Project/
│
├── data/                      # Data storage directory
├── notebooks/                 # Jupyter notebooks for analysis
├── src/                       # Core source code
│   ├── data_loader.py        # Market data fetching and processing
│   ├── risk_engine.py        # Risk calculation engine
│   ├── portfolio.py          # Portfolio management
│   ├── ml_model.py           # Machine learning risk prediction
│   └── utils.py              # Helper functions and utilities
│
├── dashboard/                 # Streamlit web application
│   └── app.py                # Interactive dashboard
│
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Features

### 1. Data Management (`data_loader.py`)
- Fetch real-time and historical market data via yfinance
- Support for Saudi stocks (e.g., 2222.SR, 1120.SR)
- Multiple timeframes (1mo, 3mo, 6mo, 1y, 2y, 5y)
- Data validation and cleaning
- Volume and price data management

### 2. Risk Engine (`risk_engine.py`)

#### Statistical Risk Metrics
- Daily and annualized returns calculation
- Correlation and covariance matrices
- EWMA (Exponentially Weighted Moving Average) volatility
- Rolling volatility analysis
- Skewness and kurtosis

#### Portfolio Risk Metrics
- **Parametric VaR** (Value at Risk) at 95% and 99% confidence
- **Historical VaR** using empirical distribution
- **Conditional VaR (CVaR)** - Expected Shortfall
- **Maximum Drawdown** analysis
- **Sharpe Ratio** and **Sortino Ratio**
- **Beta** calculation against market benchmark

#### Advanced Risk Analysis
- **Monte Carlo Simulation** (up to 50,000 paths)
- **Stress Testing** with customizable scenarios
- **Liquidity Risk** indicator and analysis
- **Portfolio optimization** (min variance, max Sharpe)

### 3. Portfolio Management (`portfolio.py`)
- Flexible portfolio weighting system
- Real-time portfolio valuation
- Rebalancing with transaction cost modeling
- Performance tracking and metrics
- Allocation drift detection
- Cash management (deposits/withdrawals)
- Portfolio import/export functionality

### 4. Machine Learning (`ml_model.py`)

#### Predictive Risk Modeling
- **Classification Model**: Predicts risk level (LOW/MEDIUM/HIGH)
- **Regression Model**: Forecasts future volatility

#### Feature Engineering
- Rolling volatility (10, 30, 60 days)
- EWMA volatility
- Moving averages (5, 20, 50 days)
- Momentum indicators
- Volume change analysis
- RSI (Relative Strength Index)
- Drawdown metrics

#### Model Options
- Random Forest (default)
- XGBoost (optional)
- Cross-validation support
- Feature importance analysis
- Model persistence (save/load)

### 5. Interactive Dashboard (`dashboard/app.py`)

#### Six Main Tabs:

**📊 Overview**
- Portfolio allocation visualization
- Cumulative returns comparison
- Performance metrics dashboard
- Real-time portfolio valuation

**📈 Risk Metrics**
- Interactive correlation heatmap
- EWMA volatility trends
- VaR comparison (Parametric, Historical, CVaR)
- Stress testing scenarios
- Maximum drawdown analysis

**🔮 Monte Carlo Simulation**
- Configurable simulation parameters
- Path visualization (up to 100 sample paths)
- Distribution of final portfolio values
- Monte Carlo VaR calculation

**🤖 ML Prediction**
- Train risk prediction models
- Feature importance visualization
- Real-time risk level prediction
- Model performance metrics

**🔔 Alerts**
- VaR threshold monitoring
- High correlation detection
- ML-based risk signals
- Configurable alert thresholds

**📋 Reports**
- Comprehensive risk summary
- Portfolio performance report
- Export functionality (JSON)

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
cd RMS_Project
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🎮 Usage

### Running the Dashboard

1. Navigate to the dashboard directory:
```bash
cd dashboard
```

2. Launch the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser to `http://localhost:8501`

### Using Individual Modules

#### Example: Data Loading
```python
from src.data_loader import DataLoader

# Initialize loader
loader = DataLoader(data_dir='data')

# Fetch Saudi stocks
tickers = ['2222.SR', '1120.SR', '2010.SR']
prices = loader.get_close_prices(tickers, period='1y')
returns = loader.get_returns(tickers, period='1y')
```

#### Example: Risk Analysis
```python
from src.risk_engine import RiskEngine
import numpy as np

# Initialize risk engine
risk_engine = RiskEngine(returns, confidence_levels=[0.95, 0.99])

# Calculate VaR
portfolio_value = 1000000
weights = np.array([0.4, 0.3, 0.3])

var_param = risk_engine.calculate_parametric_var(portfolio_value, weights)
var_hist = risk_engine.calculate_historical_var(portfolio_value, weights)
cvar = risk_engine.calculate_cvar(portfolio_value, weights)

print(f"Parametric VaR (95%): ${var_param[0.95]:,.2f}")
print(f"Historical VaR (95%): ${var_hist[0.95]:,.2f}")
print(f"CVaR (95%): ${cvar[0.95]:,.2f}")
```

#### Example: Portfolio Management
```python
from src.portfolio import Portfolio

# Create portfolio
portfolio = Portfolio(
    name="My Saudi Portfolio",
    initial_value=1000000,
    tickers=['2222.SR', '1120.SR', '2010.SR'],
    weights=np.array([0.4, 0.3, 0.3])
)

# Update with latest prices
latest_prices = prices.iloc[-1]
portfolio.update_prices(latest_prices)

# Get performance metrics
metrics = portfolio.get_performance_metrics(returns)
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
```

#### Example: ML Risk Prediction
```python
from src.ml_model import RiskPredictor

# Initialize predictor
predictor = RiskPredictor(model_type='classification')

# Prepare features
features = predictor.prepare_features(returns, prices, volume)
target = predictor.create_target_classification(returns, weights)

# Train model
results = predictor.train_model(features, target, use_xgboost=True)
print(f"Test Accuracy: {results['test_accuracy']:.4f}")

# Predict current risk
current_features = features.iloc[-1:]
risk_level = predictor.predict_risk_level(current_features)
print(f"Current Risk Level: {risk_level[0]}")
```

#### Example: Monte Carlo Simulation
```python
# Run Monte Carlo simulation
mc_results = risk_engine.monte_carlo_simulation(
    portfolio_value=1000000,
    weights=weights,
    num_simulations=10000,
    time_horizon=252
)

print(f"Mean Final Value: ${mc_results['mean_final_value']:,.2f}")
print(f"5th Percentile: ${mc_results['percentiles']['5th']:,.2f}")
print(f"95th Percentile: ${mc_results['percentiles']['95th']:,.2f}")
```

---

## 🌍 Market-Specific Usage

### Saudi Stock Market (Tadawul)

Top 10 Tadawul stocks included:
- 2222.SR - Saudi Aramco
- 1120.SR - Al Rajhi Bank
- 2010.SR - SABIC
- 2030.SR - Saudi Telecom (STC)
- 1210.SR - Saudi Electricity
- 1111.SR - Saudi Kayan
- 2382.SR - Etihad Etisalat (Mobily)
- 1180.SR - Almarai
- 2220.SR - Maaden
- 4030.SR - Saudi Airlines

```python
from src.utils import get_tadawul_top_stocks

tickers = get_tadawul_top_stocks()
loader = DataLoader()
prices = loader.get_close_prices(tickers, period='1y')
```

### Global Stocks

```python
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
loader = DataLoader()
prices = loader.get_close_prices(tickers, period='2y')
```

---

## 🔔 Alert System

The system triggers alerts when:

1. **VaR Threshold Exceeded**
   - Configurable threshold (default 5% of portfolio value)
   - Monitors all confidence levels

2. **High Correlation Detected**
   - Threshold: correlation > 0.7 (configurable)
   - Identifies diversification risks

3. **Liquidity Risk**
   - Low liquidity score
   - Extended liquidation timeframes

4. **ML Risk Signal**
   - Model predicts HIGH risk level
   - Based on trained predictive model

---

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

---

## 📊 Dependencies

Core dependencies:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `yfinance` - Market data fetching
- `scikit-learn` - Machine learning
- `xgboost` - Gradient boosting
- `streamlit` - Web dashboard
- `plotly` - Interactive visualizations
- `scipy` - Statistical functions
- `matplotlib`, `seaborn` - Data visualization
- `ta` - Technical analysis indicators

See `requirements.txt` for full list.

---

## 🔧 Configuration

### Dashboard Configuration
Edit `dashboard/app.py`:
- Default portfolio value
- Confidence levels
- Alert thresholds
- Visualization settings

### Risk Engine Configuration
Edit `src/risk_engine.py`:
- VaR calculation methods
- Monte Carlo parameters
- Stress test scenarios

### ML Model Configuration
Edit `src/ml_model.py`:
- Model hyperparameters
- Feature engineering parameters
- Risk level thresholds

---

## 🚀 Extending the System

### Adding New Risk Metrics

1. Add method to `RiskEngine` class in `src/risk_engine.py`:
```python
def calculate_your_metric(self, ...):
    # Implementation
    result = ...
    self.risk_metrics['your_metric'] = result
    return result
```

2. Update dashboard in `dashboard/app.py` to display the metric.

### Adding New Data Sources

1. Extend `DataLoader` class in `src/data_loader.py`:
```python
def fetch_from_new_source(self, ...):
    # Implementation
    return data
```

### Adding New ML Features

1. Update `prepare_features()` in `src/ml_model.py`:
```python
ticker_features[f'{ticker}_new_feature'] = calculation
```

---

## 📈 Future Enhancements

### Planned Features:
- ✅ Core risk metrics implementation
- ✅ ML-based risk prediction
- ✅ Interactive dashboard
- 🔄 Backtesting module
- 🔄 API layer (FastAPI)
- 🔄 Docker deployment
- 🔄 Real-time data streaming
- 🔄 Multi-currency support
- 🔄 Advanced portfolio optimization algorithms
- 🔄 Integration with broker APIs

---

## 🐳 Docker Deployment (Planned)

Future docker-compose.yml:
```yaml
version: '3.8'
services:
  rms-dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
```

---

## 📝 Best Practices

### Data Management
- Cache frequently accessed data
- Validate ticker symbols before fetching
- Handle missing data appropriately
- Store historical data for backtesting

### Risk Calculations
- Use appropriate confidence levels
- Consider market conditions in stress tests
- Regularly update correlation matrices
- Monitor rolling volatility trends

### ML Models
- Retrain models periodically
- Validate predictions against actual outcomes
- Monitor feature importance changes
- Use cross-validation for robust estimates

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Authors

Developed by professional quantitative developers and financial risk engineers.

---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check documentation
- Review example notebooks

---

## ⚠️ Disclaimer

This software is for educational and informational purposes only. It should not be considered as financial advice. Always consult with qualified financial advisors before making investment decisions. Past performance does not guarantee future results.

---

## 🎓 References

### Risk Management
- Jorion, P. (2006). Value at Risk: The New Benchmark for Managing Financial Risk
- Hull, J. C. (2018). Risk Management and Financial Institutions

### Quantitative Finance
- Fabozzi, F. J. (2010). Quantitative Equity Investing
- Grinold, R. C., & Kahn, R. N. (1999). Active Portfolio Management

### Machine Learning in Finance
- Dixon, M. F., Halperin, I., & Bilokon, P. (2020). Machine Learning in Finance
- López de Prado, M. (2018). Advances in Financial Machine Learning

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Status**: Production Ready
#   R M S - P r o j e c t -  
 #   R M S - P r o j e c t -  
 #   R M S - P r o j e c t  
 