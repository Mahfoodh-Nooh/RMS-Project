"""
API Client for the RMS FastAPI backend.

Provides a thin, typed wrapper around the backend REST API.
All functions raise `APIError` on failure with a human-readable message.
"""

from __future__ import annotations

import requests
from typing import Any, Dict, List, Optional


import os
BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/") + "/api/v1"
DEFAULT_TIMEOUT = 30


class APIError(Exception):
    """Raised when the backend returns an error or is unreachable."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _auth_headers(token: Optional[str]) -> Dict[str, str]:
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _handle_response(response: requests.Response) -> Any:
    """Parse JSON or raise APIError with a useful message."""
    try:
        data = response.json()
    except ValueError:
        raise APIError(
            f"Invalid JSON response from backend (status {response.status_code})",
            status_code=response.status_code,
        )

    if not response.ok:
        detail = data.get("detail") if isinstance(data, dict) else None
        message = detail if isinstance(detail, str) else str(detail or data)
        raise APIError(
            f"API error ({response.status_code}): {message}",
            status_code=response.status_code,
        )

    return data


def _request(
    method: str,
    path: str,
    *,
    token: Optional[str] = None,
    json_body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Any:
    url = f"{BASE_URL}{path}"
    headers = _auth_headers(token)

    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=json_body,
            params=params,
            timeout=DEFAULT_TIMEOUT,
        )
    except requests.ConnectionError:
        raise APIError("Cannot reach backend. Is FastAPI running at 127.0.0.1:8000?")
    except requests.Timeout:
        raise APIError("Backend request timed out.")
    except requests.RequestException as e:
        raise APIError(f"Request failed: {e}")

    return _handle_response(response)


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #

def register_user(email: str, full_name: str, password: str) -> Dict[str, Any]:
    return _request(
        "POST",
        "/auth/register",
        json_body={"email": email, "full_name": full_name, "password": password},
    )


def login_user(email: str, password: str) -> str:
    """Return the JWT access token."""
    data = _request(
        "POST",
        "/auth/login",
        json_body={"email": email, "password": password},
    )
    token = data.get("access_token")
    if not token:
        raise APIError("Login succeeded but no access token was returned.")
    return token


def get_current_user(token: str) -> Dict[str, Any]:
    return _request("GET", "/auth/me", token=token)


# --------------------------------------------------------------------------- #
# Portfolios
# --------------------------------------------------------------------------- #

def list_portfolios(token: str) -> List[Dict[str, Any]]:
    return _request("GET", "/portfolios", token=token)


def get_portfolio(token: str, portfolio_id: int) -> Dict[str, Any]:
    return _request("GET", f"/portfolios/{portfolio_id}", token=token)


def create_portfolio(
    token: str,
    name: str,
    initial_value: float,
    positions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return _request(
        "POST",
        "/portfolios",
        token=token,
        json_body={
            "name": name,
            "initial_value": initial_value,
            "positions": positions,
        },
    )


def add_position(
    token: str,
    portfolio_id: int,
    ticker: str,
    quantity: float,
) -> Dict[str, Any]:
    return _request(
        "POST",
        f"/portfolios/{portfolio_id}/positions",
        token=token,
        json_body={"ticker": ticker, "quantity": quantity},
    )


def delete_portfolio(token: str, portfolio_id: int) -> None:
    _request("DELETE", f"/portfolios/{portfolio_id}", token=token)


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #

def calculate_risk(
    token: str,
    portfolio_id: int,
    period: str = "1y",
    confidence_levels: Optional[List[float]] = None,
) -> Dict[str, Any]:
    return _request(
        "POST",
        "/risk/analyze",
        token=token,
        json_body={
            "portfolio_id": portfolio_id,
            "period": period,
            "confidence_levels": confidence_levels or [0.95, 0.99],
        },
    )


def stress_test(
    token: str,
    portfolio_id: int,
    period: str = "1y",
    confidence_levels: Optional[List[float]] = None,
) -> Dict[str, Any]:
    return _request(
        "POST",
        "/risk/stress-test",
        token=token,
        json_body={
            "portfolio_id": portfolio_id,
            "period": period,
            "confidence_levels": confidence_levels or [0.95, 0.99],
        },
    )


def monte_carlo(
    token: str,
    portfolio_id: int,
    period: str = "1y",
    confidence_levels: Optional[List[float]] = None,
    num_simulations: int = 1000,
    time_horizon: int = 252,
) -> Dict[str, Any]:
    return _request(
        "POST",
        "/risk/monte-carlo",
        token=token,
        json_body={
            "portfolio_id": portfolio_id,
            "period": period,
            "confidence_levels": confidence_levels or [0.95, 0.99],
        },
        params={
            "num_simulations": num_simulations,
            "time_horizon": time_horizon,
        },
    )


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #

def health_check() -> bool:
    try:
        response = requests.get("http://127.0.0.1:8000/", timeout=5)
        return response.ok
    except requests.RequestException:
        return False
