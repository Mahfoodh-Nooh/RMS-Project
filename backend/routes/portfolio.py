from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from database import get_db
import models
import schemas
import auth

router = APIRouter(prefix="/api/v1/portfolios", tags=["portfolios"])


@router.post("", response_model=schemas.PortfolioOut, status_code=status.HTTP_201_CREATED)
def create_portfolio(
    payload: schemas.PortfolioCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    portfolio = models.Portfolio(
        name=payload.name,
        initial_value=payload.initial_value,
        owner_id=current_user.id,
    )
    db.add(portfolio)
    db.flush()

    for pos in (payload.positions or []):
        db.add(models.Position(ticker=pos.ticker.upper(), quantity=pos.quantity, portfolio_id=portfolio.id))

    db.commit()
    db.refresh(portfolio)
    return portfolio


@router.get("", response_model=List[schemas.PortfolioOut])
def list_portfolios(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    return db.query(models.Portfolio).filter(models.Portfolio.owner_id == current_user.id).all()


@router.get("/{portfolio_id}", response_model=schemas.PortfolioOut)
def get_portfolio(
    portfolio_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.owner_id == current_user.id,
    ).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    return portfolio


@router.post("/{portfolio_id}/positions", response_model=schemas.PositionOut, status_code=status.HTTP_201_CREATED)
def add_position(
    portfolio_id: int,
    payload: schemas.PositionCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.owner_id == current_user.id,
    ).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")

    position = models.Position(ticker=payload.ticker.upper(), quantity=payload.quantity, portfolio_id=portfolio_id)
    db.add(position)
    db.commit()
    db.refresh(position)
    return position


@router.delete("/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_portfolio(
    portfolio_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(auth.get_current_user),
):
    portfolio = db.query(models.Portfolio).filter(
        models.Portfolio.id == portfolio_id,
        models.Portfolio.owner_id == current_user.id,
    ).first()
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    db.delete(portfolio)
    db.commit()
