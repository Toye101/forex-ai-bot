import MetaTrader5 as mt5
from datetime import datetime

# Initialize MT5 connection
mt5.initialize()

symbol = "EURUSD"
lot = 0.01  # micro lot
sl_pips = 20
tp_pips = 40

# Ensure symbol is visible in Market Watch
if not mt5.symbol_select(symbol, True):
    print(f"❌ Failed to select {symbol}")
    mt5.shutdown()
    exit()

# Get current prices
symbol_info_tick = mt5.symbol_info_tick(symbol)
if symbol_info_tick is None:
    print("❌ Failed to get symbol tick info.")
    mt5.shutdown()
    exit()

price = symbol_info_tick.ask
sl = price - sl_pips * 0.0001
tp = price + tp_pips * 0.0001

# Build order request
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": lot,
    "type": mt5.ORDER_TYPE_BUY,
    "price": price,
    "sl": sl,
    "tp": tp,
    "deviation": 10,
    "magic": 123456,
    "comment": "Test trade",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_FOK,
}

# Send the trade
result = mt5.order_send(request)

# Show the result
if result.retcode == mt5.TRADE_RETCODE_DONE:
    print(f"✅ Trade successful! Ticket: {result.order}")
    print(f"Bought {lot} {symbol} at {price}")
else:
    print(f"❌ Trade failed, retcode={result.retcode}")
    print(result)

mt5.shutdown()