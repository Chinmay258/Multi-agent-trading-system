//+------------------------------------------------------------------+
//|  TradingSystemEA.mq5                                             |
//|  HTTP polling bridge for the Python multi-agent trading system.  |
//|                                                                   |
//|  Uses MT5's built-in WebRequest() — no external libraries.       |
//|  Works with ALL MT5 builds (no minimum build requirement).        |
//|                                                                   |
//|  Architecture (reversed — EA calls Python, not the other way):   |
//|    Every 10 seconds OnTimer() fires:                              |
//|      1. POST balance + positions  →  http://python:5556/mt5/state |
//|      2. GET next command          ←  http://python:5556/mt5/command|
//|      3. Execute command (if any)                                  |
//|      4. POST result               →  http://python:5556/mt5/result|
//|                                                                   |
//|  Python side (mt5_bridge.py) runs a small aiohttp server on      |
//|  port 5556 and queues commands via asyncio.Queue.                 |
//|                                                                   |
//|  IMPORTANT — before attaching this EA you MUST add the Python     |
//|  URL to MT5's allowed WebRequest list:                            |
//|    Tools → Options → Expert Advisors → Allow WebRequest for       |
//|    listed URL  →  add  http://localhost:5556                      |
//+------------------------------------------------------------------+

#property copyright "TradingSystem Phase 7"
#property version   "1.04"

//--- Input parameters
input string PythonUrl   = "http://127.0.0.1:5556";  // Python aiohttp server URL
input int    MagicNumber = 20250101;                 // Magic number for EA orders
input bool   VerboseLog  = false;                    // Log every request/response

//--- HTTP timeout for each WebRequest call (milliseconds)
#define HTTP_TIMEOUT_MS 3000

//--- Normalised base URL (trailing slash stripped in OnInit)
string g_python_url = "";

//+------------------------------------------------------------------+
//| Expert initialisation                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   // Normalise base URL once — strip trailing slash so all path concatenations
   // produce valid URLs regardless of what the user typed in the input field.
   g_python_url = PythonUrl;
   if(StringLen(g_python_url) > 0 &&
      StringSubstr(g_python_url, StringLen(g_python_url) - 1, 1) == "/")
      g_python_url = StringSubstr(g_python_url, 0, StringLen(g_python_url) - 1);

   Print("[TradingSystemEA] Using URL: ", g_python_url);
   Print("[TradingSystemEA] URL length: ", StringLen(g_python_url));
   string test_url = g_python_url + "/mt5/state";
   Print("[TradingSystemEA] State URL will be: ", test_url);

   if(!EventSetMillisecondTimer(10000))
   {
      Print("[TradingSystemEA] ERROR: Failed to set 10-second timer");
      return INIT_FAILED;
   }
   Print("[TradingSystemEA] Initialised. Python URL=", g_python_url,
         "  Magic=", MagicNumber, "  Interval=10s");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialisation                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("[TradingSystemEA] Deinitialised. Reason=", reason);
}

//+------------------------------------------------------------------+
//| OnTimer — fires every 10 seconds                                 |
//+------------------------------------------------------------------+
void OnTimer()
{
   //--- 1. Send current account state to Python
   PostState();

   //--- 2. Poll Python for the next pending command
   string cmd_json = GetCommand();
   if(StringLen(cmd_json) == 0)
      return;

   string status     = ParseJSONString(cmd_json, "status");
   string command_id = ParseJSONString(cmd_json, "command_id");
   string action     = ParseJSONString(cmd_json, "action");

   if(status != "ok" || StringLen(action) == 0)
      return;   // empty queue or unrecognised response

   if(VerboseLog)
      Print("[TradingSystemEA] Executing: ", action, " (id=", command_id, ")");

   //--- 3. Execute the command
   string result_json = ExecuteAction(action, cmd_json, command_id);

   if(VerboseLog)
      Print("[TradingSystemEA] Result: ", result_json);

   //--- 4. POST the result back to Python
   PostResult(result_json);
}

//+------------------------------------------------------------------+
//| POST current account state (balance + positions) to Python       |
//+------------------------------------------------------------------+
void PostState()
{
   double equity     = AccountInfoDouble(ACCOUNT_EQUITY);
   double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double usedMargin = AccountInfoDouble(ACCOUNT_MARGIN);
   string currency   = AccountInfoString(ACCOUNT_CURRENCY);

   string balance_json =
      "{\"total_equity\":"  + DoubleToString(equity,     2) + ","
    + "\"free_margin\":"    + DoubleToString(freeMargin, 2) + ","
    + "\"used_margin\":"    + DoubleToString(usedMargin, 2) + ","
    + "\"currency\":\""     + currency                      + "\"}";

   string positions_json = BuildPositionsJSON();

   string body = "{\"balance\":" + balance_json
               + ",\"positions\":" + positions_json + "}";

   string url_state = g_python_url + "/mt5/state";
   PostJSON(url_state, body);
}

//+------------------------------------------------------------------+
//| GET the next pending command from Python                         |
//| Returns empty string on failure or HTTP error.                   |
//+------------------------------------------------------------------+
string GetCommand()
{
   char   post_data[];          // empty body for GET
   char   result[];
   string result_headers;
   string headers = "Connection: keep-alive\r\n";
   string cmd_url = g_python_url + "/mt5/command";

   int code = WebRequest("GET",
                         cmd_url,
                         headers,
                         HTTP_TIMEOUT_MS,
                         post_data,
                         result,
                         result_headers);
   if(code != 200)
   {
      // Always log — not gated on VerboseLog
      int    mqlErr = GetLastError();
      string mqlStr = IntegerToString(mqlErr);
      Print("[TradingSystemEA] GET failed: url=", cmd_url,
            " http=", code, " mql_err=", mqlStr,
            " (5003=URL not in allowlist  5004=timeout  5005=refused)");
      return "";
   }
   if(VerboseLog)
      Print("[TradingSystemEA] GET ok: ", cmd_url);
   return CharArrayToString(result);
}

//+------------------------------------------------------------------+
//| POST an execution result back to Python                          |
//+------------------------------------------------------------------+
void PostResult(const string &body)
{
   string url_result = g_python_url + "/mt5/result";
   PostJSON(url_result, body);
}

//+------------------------------------------------------------------+
//| Shared JSON POST helper                                           |
//| Returns true if the server replied with HTTP 200.                |
//+------------------------------------------------------------------+
bool PostJSON(const string &url, const string &body)
{
   char   data[];
   int    len = StringLen(body);
   ArrayResize(data, len);
   StringToCharArray(body, data, 0, len);

   char   result[];
   string result_headers;
   string headers = "Content-Type: application/json\r\nConnection: keep-alive\r\n";

   int code = WebRequest("POST", url, headers, HTTP_TIMEOUT_MS,
                         data, result, result_headers);
   if(code != 200)
   {
      // Always log — not gated on VerboseLog — so errors are visible in Experts tab
      int    mqlErr = GetLastError();
      string mqlStr = IntegerToString(mqlErr);
      Print("[TradingSystemEA] POST failed: url=", url,
            " http=", code, " mql_err=", mqlStr,
            " (5003=URL not in allowlist  5004=timeout  5005=refused)");
      return false;
   }
   if(VerboseLog)
      Print("[TradingSystemEA] POST ok: ", url);
   return true;
}

//+------------------------------------------------------------------+
//| Route action string to the appropriate handler                   |
//| Always injects command_id into the returned JSON.                |
//+------------------------------------------------------------------+
string ExecuteAction(const string &action,
                     const string &cmd_json,
                     const string &command_id)
{
   string result_body;

   if(action == "PLACE_ORDER")
      result_body = HandlePlaceOrder(cmd_json);
   else if(action == "CANCEL_ORDER")
      result_body = HandleCancelOrder(cmd_json);
   else if(action == "MODIFY_STOPS")
      result_body = HandleModifyStops(cmd_json);
   else if(action == "CLOSE_POSITION")
      result_body = HandleClosePosition(cmd_json);
   else
   {
      string ec_unknown = "unknown_action";
      string em_unknown = "Unknown action: " + action;
      result_body = BuildError(ec_unknown, em_unknown, 0);
   }

   // Inject command_id before the closing brace so Python can match the result
   // All handler functions return strings ending with "}"
   int last = StringLen(result_body) - 1;
   if(last > 0 && StringGetCharacter(result_body, last) == '}')
   {
      result_body = StringSubstr(result_body, 0, last)
                  + ",\"command_id\":\"" + command_id + "\"}";
   }
   return result_body;
}

//+------------------------------------------------------------------+
//| PLACE_ORDER handler                                               |
//+------------------------------------------------------------------+
string HandlePlaceOrder(const string &raw)
{
   string symbol        = ParseJSONString(raw, "symbol");
   string side          = ParseJSONString(raw, "side");
   double volume        = ParseJSONDouble(raw, "volume");
   double stopLossPct   = ParseJSONDouble(raw, "stop_loss_pct");
   double takeProfitPct = ParseJSONDouble(raw, "take_profit_pct");
   string comment       = ParseJSONString(raw, "comment");

   if(symbol == "" || volume <= 0)
   {
      string ec_inv = "invalid_params";
      string em_inv = "symbol or volume missing/invalid";
      return BuildError(ec_inv, em_inv, 0);
   }

   ENUM_ORDER_TYPE orderType = (side == "buy") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

   // Compute SL/TP from the live market price at the moment of execution.
   // This is more accurate than using the signal price sent from Python
   // (which may be several seconds old by the time the order is filled).
   int    digits  = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   double entryPx = (orderType == ORDER_TYPE_BUY)
                    ? SymbolInfoDouble(symbol, SYMBOL_ASK)
                    : SymbolInfoDouble(symbol, SYMBOL_BID);
   double slPrice = 0.0;
   double tpPrice = 0.0;

   if(stopLossPct > 0.0)
   {
      slPrice = (orderType == ORDER_TYPE_BUY)
                ? entryPx * (1.0 - stopLossPct)
                : entryPx * (1.0 + stopLossPct);
      slPrice = NormalizeDouble(slPrice, digits);
   }
   if(takeProfitPct > 0.0)
   {
      tpPrice = (orderType == ORDER_TYPE_BUY)
                ? entryPx * (1.0 + takeProfitPct)
                : entryPx * (1.0 - takeProfitPct);
      tpPrice = NormalizeDouble(tpPrice, digits);
   }

   MqlTradeRequest req = {};
   MqlTradeResult  res = {};

   req.action       = TRADE_ACTION_DEAL;
   req.symbol       = symbol;
   req.volume       = volume;
   req.type         = orderType;
   req.type_filling = ORDER_FILLING_IOC;
   req.sl           = slPrice;
   req.tp           = tpPrice;
   req.magic        = MagicNumber;
   req.comment      = comment;
   req.price        = entryPx;
   req.deviation    = 20;   // max slippage in points

   bool ok = OrderSend(req, res);
   if(!ok || res.retcode != TRADE_RETCODE_DONE)
   {
      int    err    = (int)res.retcode;
      string ec_rej = "order_rejected";
      string em_rej = res.comment;
      Print("[TradingSystemEA] OrderSend failed: retcode=", err,
            " comment=", em_rej);
      return BuildError(ec_rej, em_rej, err);
   }

   Print("[TradingSystemEA] OrderSend ok: order=", res.order,
         " entry=",  DoubleToString(entryPx, digits),
         " sl=",     DoubleToString(slPrice,  digits),
         " tp=",     DoubleToString(tpPrice,  digits));

   string orderIdStr = IntegerToString((long)res.order);
   double fillPrice  = res.price;
   double fillQty    = res.volume;

   return "{\"status\":\"ok\","
        + "\"order_id\":\""    + orderIdStr                  + "\","
        + "\"fill_price\":"    + DoubleToString(fillPrice, 8) + ","
        + "\"fill_quantity\":" + DoubleToString(fillQty,   8) + "}";
}

//+------------------------------------------------------------------+
//| CANCEL_ORDER handler                                              |
//+------------------------------------------------------------------+
string HandleCancelOrder(const string &raw)
{
   string orderIdStr = ParseJSONString(raw, "order_id");
   ulong  ticket     = (ulong)StringToInteger(orderIdStr);

   MqlTradeRequest req = {};
   MqlTradeResult  res = {};
   req.action = TRADE_ACTION_REMOVE;
   req.order  = ticket;

   bool ok = OrderSend(req, res);
   if(!ok)
   {
      int err = GetLastError();
      if(err == 4756 || err == 4108)   // not found / already closed
         return "{\"status\":\"not_found\",\"order_id\":\"" + orderIdStr + "\"}";
      string ec_can = "cancel_failed";
      string em_can = "OrderSend REMOVE failed";
      return BuildError(ec_can, em_can, err);
   }
   return "{\"status\":\"ok\",\"order_id\":\"" + orderIdStr + "\"}";
}

//+------------------------------------------------------------------+
//| MODIFY_STOPS handler                                              |
//+------------------------------------------------------------------+
string HandleModifyStops(const string &raw)
{
   string orderIdStr = ParseJSONString(raw, "order_id");
   ulong  ticket     = (ulong)StringToInteger(orderIdStr);
   double sl         = ParseJSONDouble(raw, "stop_loss");
   double tp         = ParseJSONDouble(raw, "take_profit");

   if(!PositionSelectByTicket(ticket))
   {
      int    err     = GetLastError();
      string ec_pos  = "position_not_found";
      string em_pos  = "Position not found: " + orderIdStr;
      return BuildError(ec_pos, em_pos, err);
   }

   MqlTradeRequest req = {};
   MqlTradeResult  res = {};
   req.action   = TRADE_ACTION_SLTP;
   req.position = ticket;
   req.symbol   = PositionGetString(POSITION_SYMBOL);
   req.sl       = sl;
   req.tp       = tp;

   bool ok = OrderSend(req, res);
   if(!ok || res.retcode != TRADE_RETCODE_DONE)
   {
      int    err     = (int)res.retcode;
      string ec_mod  = "modify_failed";
      string em_mod  = res.comment;
      return BuildError(ec_mod, em_mod, err);
   }
   return "{\"status\":\"ok\",\"order_id\":\"" + orderIdStr + "\"}";
}

//+------------------------------------------------------------------+
//| CLOSE_POSITION handler                                            |
//| Finds the first position matching symbol+magic and closes it.    |
//+------------------------------------------------------------------+
string HandleClosePosition(const string &raw)
{
   string symbol   = ParseJSONString(raw, "symbol");
   long   magic    = (long)ParseJSONDouble(raw, "magic");

   if(symbol == "")
   {
      string ec_cp = "invalid_params";
      string em_cp = "symbol missing";
      return BuildError(ec_cp, em_cp, 0);
   }

   int total = PositionsTotal();
   for(int i = total - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL) != symbol) continue;
      if(magic > 0 && PositionGetInteger(POSITION_MAGIC) != magic) continue;

      long   posType   = PositionGetInteger(POSITION_TYPE);
      double qty       = PositionGetDouble(POSITION_VOLUME);
      string closeStr  = (posType == POSITION_TYPE_BUY) ? "sell" : "buy";

      ENUM_ORDER_TYPE closeType = (posType == POSITION_TYPE_BUY)
                                  ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
      double closePrice = (posType == POSITION_TYPE_BUY)
                          ? SymbolInfoDouble(symbol, SYMBOL_BID)
                          : SymbolInfoDouble(symbol, SYMBOL_ASK);

      MqlTradeRequest req = {};
      MqlTradeResult  res = {};
      req.action       = TRADE_ACTION_DEAL;
      req.position     = ticket;
      req.symbol       = symbol;
      req.volume       = qty;
      req.type         = closeType;
      req.type_filling = ORDER_FILLING_IOC;
      req.price        = closePrice;
      req.deviation    = 20;
      req.magic        = magic;

      bool ok = OrderSend(req, res);
      if(!ok || res.retcode != TRADE_RETCODE_DONE)
      {
         int    err    = (int)res.retcode;
         string ec_cf  = "close_failed";
         string em_cf  = res.comment;
         Print("[TradingSystemEA] ClosePosition failed: ticket=", ticket,
               " retcode=", err, " comment=", em_cf);
         return BuildError(ec_cf, em_cf, err);
      }

      string orderIdStr = IntegerToString((long)res.order);
      double fillPrice  = res.price;
      double fillQty    = res.volume;

      Print("[TradingSystemEA] Position closed: ticket=", ticket,
            " symbol=", symbol, " side=", closeStr,
            " qty=", fillQty, " price=", fillPrice);

      return "{\"status\":\"ok\","
           + "\"order_id\":\""    + orderIdStr                  + "\","
           + "\"side\":\""        + closeStr                    + "\","
           + "\"fill_price\":"    + DoubleToString(fillPrice, 8) + ","
           + "\"fill_quantity\":" + DoubleToString(fillQty,   8) + "}";
   }

   string ec_nf = "position_not_found";
   string em_nf = "No open position for symbol: " + symbol;
   Print("[TradingSystemEA] ClosePosition: ", em_nf);
   return BuildError(ec_nf, em_nf, 0);
}

//+------------------------------------------------------------------+
//| Build the positions JSON array from currently open positions     |
//+------------------------------------------------------------------+
string BuildPositionsJSON()
{
   int    total = PositionsTotal();
   string arr   = "[";
   bool   first = true;

   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(!PositionSelectByTicket(ticket))
         continue;

      string sym     = PositionGetString(POSITION_SYMBOL);
      long   posType = PositionGetInteger(POSITION_TYPE);
      double qty     = PositionGetDouble(POSITION_VOLUME);
      double entry   = PositionGetDouble(POSITION_PRICE_OPEN);
      double cur     = PositionGetDouble(POSITION_PRICE_CURRENT);
      double pnl     = PositionGetDouble(POSITION_PROFIT);
      double sl      = PositionGetDouble(POSITION_SL);
      double tp      = PositionGetDouble(POSITION_TP);

      string sideStr = (posType == POSITION_TYPE_BUY) ? "buy" : "sell";

      if(!first)
         arr += ",";
      first = false;

      arr += "{"
           + "\"ticket\":"        + IntegerToString((long)ticket) + ","
           + "\"symbol\":\""      + sym                           + "\","
           + "\"side\":\""        + sideStr                       + "\","
           + "\"quantity\":"      + DoubleToString(qty,     8)    + ","
           + "\"entry_price\":"   + DoubleToString(entry,   8)    + ","
           + "\"current_price\":" + DoubleToString(cur,     8)    + ","
           + "\"unrealised_pnl\":" + DoubleToString(pnl,   2)    + ","
           + "\"stop_loss\":"     + DoubleToString(sl,      8)    + ","
           + "\"take_profit\":"   + DoubleToString(tp,      8)
           + "}";
   }
   arr += "]";
   return arr;
}

//+------------------------------------------------------------------+
//| Build a JSON error response                                       |
//+------------------------------------------------------------------+
string BuildError(const string &code, const string &message, int errCode)
{
   Print("[TradingSystemEA] ERROR ", code, ": ", message,
         " (code=", errCode, ")");
   return "{\"status\":\"error\","
        + "\"error_code\":"      + IntegerToString(errCode) + ","
        + "\"error_message\":\"" + EscapeJSON(message)      + "\"}";
}

//+------------------------------------------------------------------+
//| Extract a string value from a flat JSON object                   |
//|   Handles both compact ("key":"value") and spaced ("key": "value")|
//|   Returns "" if not present.                                      |
//+------------------------------------------------------------------+
string ParseJSONString(const string &json, const string &key)
{
   // Find  "key":  (colon without requiring immediate quote)
   string needle = "\"" + key + "\":";
   int start = StringFind(json, needle);
   if(start < 0)
      return "";
   start += StringLen(needle);

   // Skip optional whitespace between : and opening "
   int len = StringLen(json);
   while(start < len && StringGetCharacter(json, start) == ' ')
      start++;

   // Expect the opening quote of the value
   if(start >= len || StringGetCharacter(json, start) != '"')
      return "";
   start++;   // consume the opening quote

   int end = StringFind(json, "\"", start);
   if(end < 0)
      return "";
   return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
//| Extract a numeric value from a flat JSON object                  |
//|   Finds  "key":number  and returns the number as double.          |
//|   Returns 0.0 if not present.                                     |
//+------------------------------------------------------------------+
double ParseJSONDouble(const string &json, const string &key)
{
   string needle = "\"" + key + "\":";
   int start = StringFind(json, needle);
   if(start < 0)
      return 0.0;
   start += StringLen(needle);

   int len = StringLen(json);
   while(start < len && StringGetCharacter(json, start) == ' ')
      start++;

   int end = start;
   while(end < len)
   {
      ushort ch = StringGetCharacter(json, end);
      if(ch == ',' || ch == '}' || ch == ']' || ch == ' ' || ch == '\n' || ch == '\r')
         break;
      end++;
   }
   if(end == start)
      return 0.0;

   return StringToDouble(StringSubstr(json, start, end - start));
}

//+------------------------------------------------------------------+
//| Escape special characters for safe embedding in JSON strings     |
//+------------------------------------------------------------------+
string EscapeJSON(const string &s)
{
   string r = s;
   StringReplace(r, "\\", "\\\\");
   StringReplace(r, "\"", "\\\"");
   StringReplace(r, "\n", "\\n");
   StringReplace(r, "\r", "\\r");
   StringReplace(r, "\t", "\\t");
   return r;
}
//+------------------------------------------------------------------+
