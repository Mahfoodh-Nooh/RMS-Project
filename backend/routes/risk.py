import sys
import os
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import get_db
import models
import schemas
import auth

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from data_loader import DataLoader
from risk_engine import RiskEngine

router = APIRouter(prefix="/api/v1/risk", tags=["risk"])

_loader = DataLoader(data_dir=os.path.join(os.path.dirname(__file__), "..", "..", "data"))


def _get_portfolio_or_404(portfolio_id: int, user_id: int, db: Session) -> models.Portfolio:
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.owner_id == user_id,
    ).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    if not portfolio.positions:
        raise HTTPException(status_code=400, detail="Portfolio has no positions")
    return portfolio


@router.post("/analyze", response_model=schemas.RiskOut)
def analyze_risk(
    payload: schemas.RiskRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    portfolio = _get_portfolio_or_404(payload.portfolio_id, current_user.id, db)

    tickers = [p.ticker for p in portfolio.positions]
    quantities = [p.quantity for p in portfolio.positions]

    try:
        returns = _loader.get_returns(tickers, period=payload.period)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Market data fetch failed: {e}")

    available = [t for t in tickers if t in returns.columns]
    if not available:
        raise HTTPException(status_code=400, detail="No valid tickers with return data")

    returns = returns[available]
    qty = np.array([quantities[tickers.index(t)] for t in available], dtype=float)
    weights = qty / qty.sum()

    engine = RiskEngine(returns, confidence_levels=payload.confidence_levels)

    stats = engine.calculate_returns_stats()
    ann_vol = float(stats["annualized_volatility"].mean())
    sharpe = float(engine.calculate_sharpe_ratio(weights))
    var_param = {str(k): round(v, 2) for k, v in engine.calculate_parametric_var(portfolio.initial_value, weights).items()}
    var_hist = {str(k): round(v, 2) for k, v in engine.calculate_historical_var(portfolio.initial_value, weights).items()}
    cvar = {str(k): round(v, 2) for k, v in engine.calculate_cvar(portfolio.initial_value, weights).items()}

    return schemas.RiskOut(
        portfolio_id=portfolio.id,
        tickers=available,
        annualized_volatility=round(ann_vol, 4),
        sharpe_ratio=round(sharpe, 4),
        parametric_var=var_param,
        historical_var=var_hist,
        cvar=cvar,
    )


@router.post("/stress-test")
def stress_test(
    payload: schemas.RiskRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    portfolio = _get_portfolio_or_404(payload.portfolio_id, current_user.id, db)

    tickers = [p.ticker for p in portfolio.positions]
    quantities = [p.quantity for p in portfolio.positions]

    try:
        returns = _loader.get_returns(tickers, period=payload.period)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Market data fetch failed: {e}")

    available = [t for t in tickers if t in returns.columns]
    returns = returns[available]
    qty = np.array([quantities[tickers.index(t)] for t in available], dtype=float)
    weights = qty / qty.sum()

    engine = RiskEngine(returns, confidence_levels=payload.confidence_levels)
    results = engine.stress_test(portfolio.initial_value, weights)

    return {
        scenario: {
            "shocked_value": round(v["shocked_value"], 2),
            "loss": round(v["loss"], 2),
            "loss_pct": round(v["loss_percentage"], 2),
        }
        for scenario, v in results.items()
    }


@router.post("/monte-carlo")
def monte_carlo(
    payload: schemas.RiskRequest,
    num_simulations: int = 1000,
    time_horizon: int = 252,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    portfolio = _get_portfolio_or_404(payload.portfolio_id, current_user.id, db)

    tickers = [p.ticker for p in portfolio.positions]
    quantities = [p.quantity for p in portfolio.positions]

    try:
        returns = _loader.get_returns(tickers, period=payload.period)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Market data fetch failed: {e}")

    available = [t for t in tickers if t in returns.columns]
    returns = returns[available]
    qty = np.array([quantities[tickers.index(t)] for t in available], dtype=float)
    weights = qty / qty.sum()

    engine = RiskEngine(returns, confidence_levels=payload.confidence_levels)
    mc = engine.monte_carlo_simulation(portfolio.initial_value, weights, num_simulations, time_horizon)

    return {
        "mean_final_value": round(mc["mean_final_value"], 2),
        "std_final_value": round(mc["std_final_value"], 2),
        "percentiles": {k: round(v, 2) for k, v in mc["percentiles"].items()},
        "mc_var": {str(k): round(v, 2) for k, v in mc["mc_var"].items()},
    }
