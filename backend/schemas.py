from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str
    created_at: datetime

    class Config:
        from_attributes = True


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class PositionCreate(BaseModel):
    ticker: str
    quantity: float


class PositionOut(BaseModel):
    id: int
    ticker: str
    quantity: float

    class Config:
        from_attributes = True


class PortfolioCreate(BaseModel):
    name: str
    initial_value: float = 1_000_000.0
    positions: Optional[List[PositionCreate]] = []


class PortfolioOut(BaseModel):
    id: int
    name: str
    initial_value: float
    created_at: datetime
    positions: List[PositionOut] = []

    class Config:
        from_attributes = True


class RiskRequest(BaseModel):
    portfolio_id: int
    period: str = "1y"
    confidence_levels: List[float] = [0.95, 0.99]


class RiskOut(BaseModel):
    portfolio_id: int
    tickers: List[str]
    annualized_volatility: float
    sharpe_ratio: float
    parametric_var: dict
    historical_var: dict
    cvar: dict
