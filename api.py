from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import DataLoader
from risk_engine import RiskEngine
from portfolio import Portfolio

app = FastAPI(
    title="Risk Management System API",
    description="Professional RMS API for equity portfolio risk analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

loader = DataLoader(data_dir='data')


class PortfolioRequest(BaseModel):
    tickers: List[str]
    weights: Optional[List[float]] = None
    portfolio_value: float = 1000000
    period: str = "1y"
    confidence_levels: List[float] = [0.95, 0.99]


@app.get("/")
def root():
    return {"status": "ok", "system": "Risk Management System API v1.0"}


@app.post("/api/v1/risk/var")
def calculate_var(request: PortfolioRequest):
    try:
        returns = loader.get_returns(request.tickers, period=request.period)

        weights = (
            np.array(request.weights) / sum(request.weights)
            if request.weights
            else np.array([1.0 / len(request.tickers)] * len(request.tickers))
        )

        engine = RiskEngine(returns, confidence_levels=request.confidence_levels)
        var_param = engine.calculate_parametric_var(request.portfolio_value, weights)
        var_hist = engine.calculate_historical_var(request.portfolio_value, weights)
        cvar = engine.calculate_cvar(request.portfolio_value, weights)

        return {
            "parametric_var": {str(k): round(v, 2) for k, v in var_param.items()},
            "historical_var": {str(k): round(v, 2) for k, v in var_hist.items()},
            "cvar": {str(k): round(v, 2) for k, v in cvar.items()}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/risk/metrics")
def get_risk_metrics(request: PortfolioRequest):
    try:
        returns = loader.get_returns(request.tickers, period=request.period)

        weights = (
            np.array(request.weights) / sum(request.weights)
            if request.weights
            else np.array([1.0 / len(request.tickers)] * len(request.tickers))
        )

        engine = RiskEngine(returns, confidence_levels=request.confidence_levels)
        stats = engine.calculate_returns_stats()
        sharpe = engine.calculate_sharpe_ratio(weights)
        sortino = engine.calculate_sortino_ratio(weights)

        return {
            "annualized_return": round(float(stats['annualized_return'].mean()), 4),
            "annualized_volatility": round(float(stats['annualized_volatility'].mean()), 4),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/risk/stress-test")
def run_stress_test(request: PortfolioRequest):
    try:
        returns = loader.get_returns(request.tickers, period=request.period)

        weights = (
            np.array(request.weights) / sum(request.weights)
            if request.weights
            else np.array([1.0 / len(request.tickers)] * len(request.tickers))
        )

        engine = RiskEngine(returns, confidence_levels=request.confidence_levels)
        results = engine.stress_test(request.portfolio_value, weights)

        return {
            scenario: {
                "shocked_value": round(v["shocked_value"], 2),
                "loss": round(v["loss"], 2),
                "loss_percentage": round(v["loss_percentage"], 2)
            }
            for scenario, v in results.items()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/risk/monte-carlo")
def run_monte_carlo(request: PortfolioRequest, num_simulations: int = 5000, time_horizon: int = 252):
    try:
        returns = loader.get_returns(request.tickers, period=request.period)

        weights = (
            np.array(request.weights) / sum(request.weights)
            if request.weights
            else np.array([1.0 / len(request.tickers)] * len(request.tickers))
        )

        engine = RiskEngine(returns, confidence_levels=request.confidence_levels)
        mc = engine.monte_carlo_simulation(request.portfolio_value, weights, num_simulations, time_horizon)

        return {
            "mean_final_value": round(mc["mean_final_value"], 2),
            "std_final_value": round(mc["std_final_value"], 2),
            "percentiles": {k: round(v, 2) for k, v in mc["percentiles"].items()},
            "mc_var": {str(k): round(v, 2) for k, v in mc["mc_var"].items()}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/market/tadawul")
def get_tadawul_stocks():
    from utils import get_tadawul_top_stocks
    return {"tickers": get_tadawul_top_stocks()}
