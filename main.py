import yfinance as yf
import numpy as np
from src.risk_engine import RiskEngine

# الأسهم
symbols = ["2222.SR", "1120.SR", "2010.SR", "7010.SR", "1180.SR"]
weights = np.array([0.25, 0.20, 0.20, 0.20, 0.15])

# تحميل البيانات
data = yf.download(symbols, start="2023-01-01")

# ❌ لا تستخدم Adj Close
data = data["Close"]

# ✅ تحويل إلى returns
returns = data.pct_change().dropna()

# تشغيل المحرك
engine = RiskEngine(returns, weights)

# ✅ اطبع فقط الموجود فعليًا
print("VaR:", engine.parametric_var())
print("CVaR:", engine.cvar())
print("Max Drawdown:", engine.max_drawdown())