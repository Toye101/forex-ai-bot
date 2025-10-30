import MetaTrader5 as mt5

# Connect to MetaTrader 5
if not mt5.initialize():
    print("❌ MT5 initialization failed")
    quit()

# Get all open positions
positions = mt5.positions_get()

if positions is None:
    print("❌ No connection or data retrieval error")
elif len(positions) == 0:
    print("ℹ No open positions at the moment.")
else:
    print(f"✅ Found {len(positions)} open position(s):\n")
    for pos in positions:
        print(f"Ticket: {pos.ticket}")
        print(f"Symbol: {pos.symbol}")
        print(f"Type: {'BUY' if pos.type == 0 else 'SELL'}")
        print(f"Volume: {pos.volume}")
        print(f"Open Price: {pos.price_open}")
        print(f"Current Price: {pos.price_current}")
        print(f"Profit: ${pos.profit:.2f}")
        print("-" * 40)

mt5.shutdown()