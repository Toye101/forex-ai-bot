import MetaTrader5 as mt5
import time

# --- Step 1: Connect to MetaTrader 5 ---
if not mt5.initialize():
    print("❌ Failed to initialize MT5")
    quit()

# --- Step 2: Get open positions ---
positions = mt5.positions_get()

if not positions:
    print("ℹ No open positions to close.")
    mt5.shutdown()
    quit()

print(f"✅ Found {len(positions)} open position(s).")

# --- Step 3: Loop through positions ---
for pos in positions:
    symbol = pos.symbol
    ticket = pos.ticket
    volume = pos.volume
    position_type = pos.type  # 0=BUY, 1=SELL

    # Ensure the symbol is selected
    mt5.symbol_select(symbol, True)
    tick = mt5.symbol_info_tick(symbol)

    if not tick:
        print(f"❌ No market data for {symbol}. Skipping.")
        continue

    # Determine trade direction and price
    if position_type == mt5.ORDER_TYPE_BUY:
        close_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
    else:
        close_type = mt5.ORDER_TYPE_BUY
        price = tick.ask

    # --- Try all safe fill types ---
    fill_types = [
        mt5.ORDER_FILLING_RETURN,
        mt5.ORDER_FILLING_IOC,
        mt5.ORDER_FILLING_FOK
    ]

    success = False
    for fill_type in fill_types:
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "Auto-close test",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": fill_type,
        }

        result = mt5.order_send(request)
        time.sleep(0.5)

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Closed {symbol} position (ticket {ticket}) successfully with fill_type={fill_type}.")
            success = True
            break
        else:
            print(f"⚠ Fill type {fill_type} failed (retcode={result.retcode}, comment={result.comment}). Trying next...")

    if not success:
        print(f"❌ Could not close {symbol} position (ticket {ticket}) with any fill type.")

mt5.shutdown()