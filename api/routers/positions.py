"""
api/routers/positions.py
-------------------------
Portfolio and trade history endpoints.

Endpoints:
    GET /positions              — current open positions from Redis
    GET /positions/balance      — current account balance from Redis
    GET /positions/history      — last 50 ExecutionResults from Redis list

Paper mode — keys written by PaperBroker (no TTL, survive restarts):
    paper_portfolio:cash        — raw Decimal string, e.g. "9799.81"
    paper_portfolio:positions   — JSON object keyed by symbol

MT5 mode — keys written by MT5Bridge on every heartbeat (TTL 30 s):
    mt5:balance                 — JSON dict: total_equity_usd, free_margin_usd, …
    mt5:positions               — JSON array of position dicts (Python symbols)

Trade history (both modes):
    execution:history           — Redis list of ExecutionResult JSON strings
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Request

from core.config import get_settings
from core.logging import get_logger

logger = get_logger("api.positions")

router = APIRouter(prefix="/positions", tags=["positions"])

_PAPER_CASH_KEY = "paper_portfolio:cash"
_PAPER_POSITIONS_KEY = "paper_portfolio:positions"
_MT5_BALANCE_KEY = "mt5:balance"
_MT5_POSITIONS_KEY = "mt5:positions"
_HISTORY_KEY = "execution:history"
_HISTORY_PAGE_SIZE = 50


@router.get("")
async def get_positions(request: Request) -> dict:
    """
    Return current open positions.

    MT5 mode: reads mt5:positions (written by MT5Bridge on each heartbeat, TTL 30 s).
    Paper mode: reads paper_portfolio:positions (written by PaperBroker, no TTL).
    Returns an empty list if the key is missing or broker not started.
    """
    bus = getattr(request.app.state, "bus", None)
    positions: list = []
    is_mt5 = get_settings().execution_broker.lower() == "mt5"

    if bus and bus._pool is not None:  # noqa: SLF001
        try:
            if is_mt5:
                raw = await bus._pool.get(_MT5_POSITIONS_KEY)  # noqa: SLF001
                if raw:
                    raw_positions: list = json.loads(raw)
                    positions = [
                        {
                            "symbol": p["symbol"],
                            "side": p["side"],
                            "quantity": float(p["quantity"]),
                            "entry_price": float(p["entry_price"]),
                            "current_price": float(p.get("current_price", p["entry_price"])),
                            "unrealised_pnl_usd": float(p.get("unrealised_pnl_usd", 0)),
                            "cost_usd": float(p["quantity"]) * float(p["entry_price"]),
                        }
                        for p in raw_positions
                    ]
            else:
                raw = await bus._pool.get(_PAPER_POSITIONS_KEY)  # noqa: SLF001
                if raw:
                    positions_data: dict = json.loads(raw)
                    positions = [
                        {
                            "symbol": p["symbol"],
                            "side": p["side"],
                            "quantity": float(p["quantity"]),
                            "entry_price": float(p["entry_price"]),
                            "current_price": float(p["entry_price"]),
                            "unrealised_pnl_usd": 0.0,
                            "cost_usd": float(p["cost_usd"]),
                        }
                        for p in positions_data.values()
                    ]
        except Exception as exc:
            logger.error("positions_read_failed", error=str(exc))

    return {"positions": positions, "count": len(positions)}


@router.get("/balance")
async def get_balance(request: Request) -> dict:
    """
    Return current account balance.

    MT5 mode: reads mt5:balance (written by MT5Bridge on each heartbeat, TTL 30 s).
    Paper mode: computed from paper_portfolio:cash + paper_portfolio:positions.
    """
    bus = getattr(request.app.state, "bus", None)
    balance = {
        "total_equity_usd": 0.0,
        "free_margin_usd": 0.0,
        "used_margin_usd": 0.0,
        "currency": "USD",
    }
    is_mt5 = get_settings().execution_broker.lower() == "mt5"

    if bus and bus._pool is not None:  # noqa: SLF001
        try:
            if is_mt5:
                raw = await bus._pool.get(_MT5_BALANCE_KEY)  # noqa: SLF001
                if raw is None:
                    logger.warning("mt5_balance_key_missing")
                    return balance
                data: dict = json.loads(raw)
                balance = {
                    "total_equity_usd": float(data.get("total_equity_usd", 0)),
                    "free_margin_usd": float(data.get("free_margin_usd", 0)),
                    "used_margin_usd": float(data.get("used_margin_usd", 0)),
                    "currency": data.get("currency", "USD"),
                }
            else:
                cash_raw = await bus._pool.get(_PAPER_CASH_KEY)  # noqa: SLF001
                if cash_raw is None:
                    logger.warning("paper_portfolio_keys_missing")
                    return balance

                free_cash = float(cash_raw)
                used_margin = 0.0

                positions_raw = await bus._pool.get(_PAPER_POSITIONS_KEY)  # noqa: SLF001
                if positions_raw:
                    positions_data: dict = json.loads(positions_raw)
                    used_margin = sum(float(p.get("cost_usd", 0)) for p in positions_data.values())

                balance = {
                    "total_equity_usd": free_cash + used_margin,
                    "free_margin_usd": free_cash,
                    "used_margin_usd": used_margin,
                    "currency": "USD",
                }
        except Exception as exc:
            logger.error("balance_read_failed", error=str(exc))

    return balance


@router.get("/history")
async def get_history(request: Request) -> dict:
    """
    Return the last 50 execution results.

    Data is read from the Redis list maintained by ExecutionAgent. Each entry
    is the raw JSON of an ExecutionResult as published to the bus. Results are
    newest-first (LPUSH means index 0 is the latest fill).
    """
    bus = getattr(request.app.state, "bus", None)
    results: list = []

    if bus and bus._pool is not None:  # noqa: SLF001
        try:
            raw_entries = await bus._pool.lrange(  # noqa: SLF001
                _HISTORY_KEY, 0, _HISTORY_PAGE_SIZE - 1
            )
            for entry in raw_entries:
                try:
                    results.append(json.loads(entry))
                except Exception:
                    pass
        except Exception as exc:
            logger.error("history_read_failed", error=str(exc))

    return {"results": results, "count": len(results)}
