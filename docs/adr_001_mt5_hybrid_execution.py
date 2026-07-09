r"""
ARCHITECTURE DECISION RECORD — ADR-001
=======================================
Title:   Hybrid Execution Architecture — Python + MetaTrader 5
Status:  ACCEPTED (MT5 integration DEFERRED — not yet implemented)
Date:    2025

Context
-------
The trading system is a Python-first autonomous multi-agent platform.
Python handles all intelligence: market analysis, signal generation,
risk management, and decision-making.

However, Python has known weaknesses as a pure execution terminal:
- Broker connectivity is fragmented (each broker needs its own adapter)
- MT5 has native, battle-tested connectivity to hundreds of brokers
- MT5 handles order lifecycle, partial fills, and requotes natively
- MT5's MQL5 environment is purpose-built for high-reliability execution
- MetaTrader is the industry standard for forex/CFD/futures execution

Decision
--------
Adopt a HYBRID architecture:

  [ Python Multi-Agent System ]  ←→  [ ZeroMQ Bridge ]  ←→  [ MT5 Terminal ]
       Intelligence Layer               IPC Layer              Execution Layer

Python remains the brain. MT5 becomes the hands.

Communication Protocol: ZeroMQ (ZMQ)
- Pattern: REQ/REP for order commands (synchronous, reliable)
- Pattern: PUB/SUB for MT5 → Python market data (optional, supplementary)
- Serialisation: JSON (human-readable, debuggable, language-agnostic)
- Transport: TCP localhost (same machine) or LAN (separate machines)

Why ZeroMQ over REST/gRPC:
- Sub-millisecond latency on localhost
- No HTTP overhead
- Battle-tested for financial systems
- Existing MQL5 ZMQ libraries (DWX Connect, ZeroMQ-MQL5)
- Broker-neutral: same Python interface regardless of which MT5 broker

Architecture Diagram
--------------------

  ┌─────────────────────────────────────────────────────┐
  │                  Python System                       │
  │                                                     │
  │  MarketDataAgent → TA Agent → Decision Agent        │
  │         ↓                          ↓                │
  │  SentimentAgent              Risk Agent             │
  │                                    ↓                │
  │                           Execution Agent           │
  │                                    │                │
  │                    ┌───────────────┘                │
  │                    ↓                                │
  │           ExecutionBroker (interface)               │
  │          /                        \                 │
  │  PaperExchange              MT5Bridge (future)      │
  │  (mock fills)          ZMQ REQ/REP socket           │
  └───────────────────────────┬─────────────────────────┘
                              │ ZeroMQ TCP
                              ↓
  ┌───────────────────────────────────────────────────┐
  │              MetaTrader 5 Terminal                 │
  │                                                   │
  │  Expert Advisor (MQL5)                            │
  │  ├── ZMQ listener (receives Python commands)      │
  │  ├── Order placement (broker native API)          │
  │  ├── Position management                         │
  │  ├── Stop loss / take profit management          │
  │  └── Execution callbacks → Python                │
  └───────────────────────────────────────────────────┘
                              │
                    Broker (ECN/STP/MM)


Implementation Plan (Future Phases)
------------------------------------
Phase 6A: MT5Bridge stub
  - Implement MT5Bridge class in agents/execution/mt5_bridge.py
  - Conforms to ExecutionBroker interface (identical to PaperExchange/LiveExchange)
  - ZMQ socket management, message serialisation, timeout handling

Phase 6B: MQL5 Expert Advisor
  - Write MT5 EA in MQL5 (separate repo: trading_system_mt5/)
  - ZMQ listener loop, order execution, position reporting
  - Test with MT5 demo account

Phase 6C: Integration
  - Wire MT5Bridge into ExecutionAgent via config: EXECUTION_BROKER=mt5
  - End-to-end paper → MT5 demo → MT5 live progression

Message Protocol (reserved, not yet implemented)
-------------------------------------------------
Python → MT5 (REQ):
{
    "action": "PLACE_ORDER",          // PLACE_ORDER | CANCEL_ORDER | CLOSE_POSITION | GET_POSITIONS
    "proposal_id": "<uuid>",
    "symbol": "BTCUSD",               // MT5 symbol format (no slash)
    "side": "buy",
    "order_type": "market",
    "quantity": 0.01,
    "stop_loss": 41000.0,
    "take_profit": 44000.0,
    "magic": 20250101                 // EA identifier
}

MT5 → Python (REP):
{
    "status": "ok",                   // ok | error | rejected
    "order_id": "123456789",
    "fill_price": 42150.50,
    "fill_quantity": 0.01,
    "timestamp": "2025-01-01T12:00:00Z",
    "error_code": null,
    "error_message": null
}

Impact on Current Design
------------------------
The following abstractions are already in place to support this:

1. ExecutionBroker interface (agents/execution/broker_interface.py)
   All execution adapters (Paper, Live, MT5) implement this interface.
   ExecutionAgent never imports PaperExchange or MT5Bridge directly —
   it receives a broker via dependency injection.

2. Generic order models (core/models/trade.py)
   OrderSide, OrderType, ExecutionResult use generic terms, not
   exchange-specific fields. MT5 adapter translates these.

3. Symbol normalisation
   MT5 uses "BTCUSD"; CCXT uses "BTC/USDT". A symbol mapper
   will live in agents/execution/symbol_mapper.py (future).

4. Config flag (planned): EXECUTION_BROKER = paper | live | mt5
   ExecutionAgent selects the adapter at startup based on this flag.

What NOT to do
--------------
- Do NOT couple ExecutionAgent to any specific broker API.
- Do NOT use ccxt order-placement methods as the canonical interface.
- Do NOT hardcode symbol formats (always normalise through symbol_mapper).
- Do NOT implement MT5 until Phase 6 — focus on stable Python infra first.

References
----------
- DWX Connect (Python ↔ MT5 ZMQ): https://github.com/darwinex/dwx-connect
- ZeroMQ MQL5 library: https://github.com/dingmaotu/mql-zmq
- CCXT unified API (current execution): https://docs.ccxt.com
"""

# This file is documentation only — no executable code.
# Import this module in tests to verify the interface contract is maintained.

ARCHITECTURE_VERSION = "ADR-001"
MT5_INTEGRATION_STATUS = "DEFERRED"
CURRENT_EXECUTION_ADAPTER = "paper"  # paper | live | mt5
