import MetaTrader5 as mt5
from pprint import pprint

# Import your saved settings
from mt5_config import MT5_SETTINGS

print("🚀 Connecting to MetaTrader 5 using mt5_config.py...")

# Initialize connection
if not mt5.initialize(login=MT5_SETTINGS["login"],
                      password=MT5_SETTINGS["password"],
                      server=MT5_SETTINGS["server"]):
    print(f"❌ Connection failed: {mt5.last_error()}")
    quit()

# Print account info
account_info = mt5.account_info()
if account_info:
    print("✅ Connected successfully!")
    print(f"Account ID : {account_info.login}")
    print(f"Balance    : ${account_info.balance}")
    print(f"Leverage   : {account_info.leverage}")
    print(f"Server     : {MT5_SETTINGS['server']}")
else:
    print("⚠ Failed to retrieve account info.")

# Check symbol info
symbol = MT5_SETTINGS["symbol"]
symbol_info = mt5.symbol_info(symbol)

if symbol_info is None:
    print(f"❌ Symbol {symbol} not found.")
else:
    print(f"\n✅ Symbol info for {symbol}:")
    print(f"Trade mode : {symbol_info.trade_mode}")
    print(f"Min lot    : {symbol_info.volume_min}")
    print(f"Max lot    : {symbol_info.volume_max}")
    print(f"Lot step   : {symbol_info.volume_step}")

# Shut down connection
mt5.shutdown()
print("\n👋 Connection closed cleanly.")