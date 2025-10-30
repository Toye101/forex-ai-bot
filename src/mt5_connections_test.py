import MetaTrader5 as mt5

# Try to connect
if not mt5.initialize():
    print("❌ MT5 connection failed:", mt5.last_error())
else:
    print("✅ Connected to MT5!")

    # Show account info
    account_info = mt5.account_info()
    if account_info:
        print(f"Account ID: {account_info.login}")
        print(f"Balance: {account_info.balance}")
        print(f"Leverage: {account_info.leverage}")
        print(f"Currency: {account_info.currency}")

    mt5.shutdown()