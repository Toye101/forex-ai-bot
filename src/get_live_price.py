import MetaTrader5 as mt5
from datetime import datetime

# Initialize MT5
if not mt5.initialize():
    print("❌ MT5 connection failed:", mt5.last_error())
    quit()

symbol = "EURUSD"
symbol_info = mt5.symbol_info(symbol)

if symbol_info is None:
    print(f"❌ {symbol} not found, check market watch visibility in MT5")
    mt5.shutdown()
    quit()

# Make sure symbol is enabled in Market Watch
if not symbol_info.visible:
    mt5.symbol_select(symbol, True)

# Get current price
tick = mt5.symbol_info_tick(symbol)
print(f"✅ {symbol} price @ {datetime.now().strftime('%H:%M:%S')}")
print(f"   Bid: {tick.bid}")
print(f"   Ask: {tick.ask}")

mt5.shutdown()