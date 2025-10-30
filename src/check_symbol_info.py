import MetaTrader5 as mt5

mt5.initialize()
symbol = "EURUSD"

info = mt5.symbol_info(symbol)
if info is None:
    print("❌ Symbol not found. Try 'EURUSDm' or 'EURUSD.' instead.")
else:
    print("✅ Symbol found:", info.name)
    print(f"Min lot: {info.volume_min}")
    print(f"Lot step: {info.volume_step}")
    print(f"Max lot: {info.volume_max}")
    print(f"Trade mode: {info.trade_mode}")
mt5.shutdown()