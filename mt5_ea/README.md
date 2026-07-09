# MT5 Expert Advisor — TradingSystemEA

Connects the Python multi-agent trading system to a MetaTrader 5 terminal using
MT5's built-in **WebRequest()** function — **no external libraries, no minimum
build version**.

The EA (`TradingSystemEA.mq5`) polls an aiohttp HTTP server running inside
Python's `MT5Bridge` (`agents/execution/mt5_bridge.py`) once per second.

---

## Architecture

```
MT5 EA (OnTimer every 1s)          Python MT5Bridge (aiohttp on :5556)
──────────────────────────         ──────────────────────────────────────
POST /mt5/state  ──────────────→   _handle_state()   (cache balance+positions)
GET  /mt5/command ◄──────────────  _handle_command()  (dequeue next command)
  execute command
POST /mt5/result  ──────────────→  _handle_result()  (resolve asyncio.Future)
```

**Direction**: The EA calls Python, not the other way around.  
**Protocol**: HTTP JSON, one state heartbeat + one command poll per second.

**Commands dispatched via HTTP** (EA executes against broker):

| Action        | Description                                    |
|---------------|------------------------------------------------|
| `PLACE_ORDER` | Market order; EA returns fill price + ticket   |
| `CANCEL_ORDER`| Cancel a pending order by ticket               |
| `MODIFY_STOPS`| Change SL/TP on an existing position           |

`get_balance()` and `get_positions()` are served from the cached heartbeat
data — they return instantly without waiting for the EA.

---

## Prerequisites

- MetaTrader 5 terminal (any build — no minimum version required)
- Python aiohttp server (`MT5Bridge`) — aiohttp is already in `pyproject.toml`

---

## Setup — Step by Step

### 1. Copy the Expert Advisor into MT5

Copy `TradingSystemEA.mq5` into MT5's Experts folder:

```
%APPDATA%\MetaQuotes\Terminal\<TerminalID>\MQL5\Experts\TradingSystemEA.mq5
```

You can find your terminal's data directory via **File → Open Data Folder** in MT5.

### 2. Compile in MetaEditor

1. Open MetaEditor (**F4** in MT5).
2. Open `TradingSystemEA.mq5`.
3. Press **F7** to compile.
4. The output panel should show **0 errors, 0 warnings**.

### 3. Allow WebRequest for the Python URL

This is a **required** step — WebRequest will silently fail without it.

In MT5: **Tools → Options → Expert Advisors**
- ☑ **Allow WebRequest for listed URL**
- Click **+** and add: `http://localhost:5556`

### 4. Attach the EA to a chart

1. In MT5, open any chart (recommended: **BTCUSD M1**).
2. Open the Navigator panel (**Ctrl+N**).
3. Drag **TradingSystemEA** from Experts onto the chart.
4. In the EA settings dialog set the inputs:
   - **PythonUrl** = `http://localhost:5556`
   - **MagicNumber** = `20250101`
   - **VerboseLog** = `false` (set to `true` for debugging)

### 5. Enable algorithmic trading in MT5

In the EA properties dialog (**Common** tab):
- ☑ **Allow algorithmic trading**

In MT5 terminal settings (**Tools → Options → Expert Advisors**):
- ☑ **Allow algorithmic trading**

### 6. Configure the Python side

In your `.env` file:

```env
EXECUTION_BROKER=mt5
MT5_LISTEN_PORT=5556
MT5_REQUEST_TIMEOUT_MS=5000
```

### 7. Start the system

```bash
docker-compose down && docker-compose up -d
```

The `ExecutionAgent` will start the aiohttp server on port 5556 and route
approved `RiskAssessment` messages to MT5 instead of the paper broker.

---

## Troubleshooting

**EA compiles but Python logs show no heartbeats**
- Verify the EA is attached to a chart and the smiley face icon is shown
  (green arrow in the toolbar means EA is running).
- Check that the WebRequest URL was added correctly:
  Tools → Options → Expert Advisors → Allow WebRequest for listed URL.
- Enable `VerboseLog=true` in the EA inputs to see every request in the Experts tab.

**`ExchangeConnectionError: MT5 EA did not respond`**
- The EA received the command but didn't post a result within `MT5_REQUEST_TIMEOUT_MS`.
- Check the Experts tab in MT5 for errors.
- Increase `MT5_REQUEST_TIMEOUT_MS` in `.env`.
- Verify the EA can reach `http://localhost:5556` (firewall, port conflicts).

**Port 5556 already in use**
- Change `MT5_LISTEN_PORT` in `.env` to a free port.
- Update the **PythonUrl** EA input and re-attach the EA.

**Orders rejected with `TRADE_RETCODE_*` errors**
- MT5 may enforce a minimum lot size. Check the symbol spec in Market Watch
  (right-click → Specification). For crypto, typical minimum is 0.001.
- The Python risk agent's `approved_quantity` must be ≥ the minimum lot size.

**Fills on demo server use `ORDER_FILLING_FOK` not `ORDER_FILLING_IOC`**
- In `HandlePlaceOrder()`, change `req.type_filling = ORDER_FILLING_IOC;`
  to `ORDER_FILLING_FOK;` and recompile.

---

## Security Notes

- The aiohttp server binds on `0.0.0.0` (all interfaces).
  Keep port 5556 firewalled from external access — only the local MT5 terminal
  should be able to reach it.
- The EA executes orders on your broker account — always test on a
  **demo account** first.
- Never expose port 5556 to the internet.
