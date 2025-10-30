# run_mt5_live_autotrader.py
"""
Live MT5 auto-trader with:
  - polling loop (auto-run)
  - live feature recomputation from MT5 candles
  - model prediction using saved LightGBM + scaler
  - fixed-dollar risk sizing (or percent option)
  - safety: single position per symbol, kill-switch file STOP_LIVE
  - logging to ../data/results/live_trades_log.csv and equity snapshots

USAGE:
  python run_mt5_live_autotrader.py

CONFIG:
  Edit the CONFIG section below (MODEL_PATH, SYMBOL, risk settings, polling interval...)
"""
import time
import math
import random
from pathlib import Path
import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
import logging
import os

# ---------------- CONFIG ----------------
MODEL_PATH = "../data/processed/lgb_h3_model.txt"
SCALER_PATH = "../data/processed/scaler_h3.save"
FEATURE_NAMES_PATH = "../data/processed/feature_names_h3.pkl"  # optional
SYMBOL = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_D1   # daily bars for 3-day horizon model; change if your model expects other timeframe
N_BARS = 500                   # how many historical bars to request to compute features reliably
POLL_INTERVAL_SECONDS = 60     # loop sleep time between checks
STOP_FILE = Path("STOP_LIVE")  # create this file to gracefully stop the bot

# Execution / risk
MODE = "fixed_dollars"         # "fixed_dollars" or "percent_of_balance"
FIXED_RISK_DOLLARS = 5.0       # $ risk per trade (you asked for $5)
RISK_PERCENT = 0.2             # fallback percent risk if MODE="percent_of_balance"
REWARD_TO_RISK = 3.0
NOLEVERAGE = True              # logical: we compute lot to achieve fixed $ risk via lots
PIP = 0.0001                   # pip unit for EURUSD
PIP_VALUE_PER_LOT = 10.0       # USD per pip per 1.0 lot (EURUSD typical)

# Broker realism
SPREAD = 0.0002                # approx spread
SLIPPAGE_MAX = 0.00025
COMMISSION = 0.00005           # applied as dollar proportion of position (simplified)

# Safety and trade rules
ONLY_ONE_POSITION_PER_SYMBOL = True
ORDER_FILLING = mt5.ORDER_FILLING_FOK if hasattr(mt5, "ORDER_FILLING_FOK") else 0
ORDER_TIME = mt5.ORDER_TIME_GTC if hasattr(mt5, "ORDER_TIME_GTC") else 0

# Logging / outputs
LOG_PATH = Path("../data/results/live_trades_log.csv")
EQUITY_LOG = Path("../data/results/equity_log.csv")
LOG_LEVEL = logging.INFO

# ---------------- helper utils ----------------
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
def human_time():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def load_model_and_scaler():
    if not Path(MODEL_PATH).exists() or not Path(SCALER_PATH).exists():
        raise FileNotFoundError("Model or scaler not found. Run forex_bot_v2.py first.")
    model = lgb.Booster(model_file=MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feat_list = None
    if Path(FEATURE_NAMES_PATH).exists():
        feat_list = joblib.load(FEATURE_NAMES_PATH)
        feat_list = [c.lower().strip() for c in feat_list]
    else:
        try:
            feat_list = [c.lower().strip() for c in model.feature_name()]
        except Exception:
            feat_list = None
    return model, scaler, feat_list

def connect_mt5():
    if not mt5.initialize():
        raise ConnectionError(f"MT5 initialize failed: {mt5.last_error()}")
    # ensure symbol is enabled
    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(f"Failed to enable symbol {SYMBOL}")
    logging.info("Connected to MT5 and selected symbol %s", SYMBOL)

def disconnect_mt5():
    try:
        mt5.shutdown()
        logging.info("MT5 shutdown cleanly.")
    except Exception:
        pass

def fetch_candles(n=N_BARS, timeframe=TIMEFRAME):
    # copy last n bars (most recent)
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError("Failed to fetch bars from MT5.")
    df = pd.DataFrame(rates)
    # convert time column to datetime if present
    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"], unit="s")
    # standard columns: open high low close tick_volume
    df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)
    return df

# Technical indicators (lightweight implementations)
def SMA(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def EMA(series, span):
    return series.ewm(span=span, adjust=False).mean()

def RSI(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=period-1, adjust=False).mean()
    ma_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def stochastic_k_d(df, k_period=14, d_period=3):
    low_min = df['low'].rolling(window=k_period, min_periods=1).min()
    high_max = df['high'].rolling(window=k_period, min_periods=1).max()
    k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-12)
    d = k.rolling(window=d_period, min_periods=1).mean()
    return k, d

def MACD(series, span_short=12, span_long=26, span_signal=9):
    ema_short = EMA(series, span_short)
    ema_long = EMA(series, span_long)
    macd = ema_short - ema_long
    signal = EMA(macd, span_signal)
    hist = macd - signal
    return macd, signal, hist

def ATR(df, period=14):
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr

def build_live_features(df):
    # df expected to have columns: open, high, low, close
    X = df.copy()
    X['return'] = X['close'].pct_change() * 100
    X['sma_10'] = SMA(X['close'], 10)
    X['sma_30'] = SMA(X['close'], 30)
    X['sma_diff'] = X['sma_10'] - X['sma_30']
    X['rsi_14'] = RSI(X['close'], 14)
    k, d = stochastic_k_d(X, 14, 3)
    X['stoch_k'] = k
    X['stoch_d'] = d
    X['ema_20'] = EMA(X['close'], 20)
    X['ema_50'] = EMA(X['close'], 50)
    macd, macd_signal, macd_hist = MACD(X['close'])
    X['macd'] = macd
    X['macd_signal'] = macd_signal
    X['macd_hist'] = macd_hist
    X['bb_width'] = (X['close'].rolling(20).max() - X['close'].rolling(20).min()) / (X['close'] + 1e-12)
    X['atr_14'] = ATR(X, 14)
    X['volatility_20'] = X['return'].rolling(20).std()
    X['regime'] = np.where(X['volatility_20'] > X['volatility_20'].median(), 1, 0)
    # add some hybrid combos used earlier
    X['trend_vol_combo'] = X['sma_diff'] * X['volatility_20']
    X['rsi_macd_combo'] = X['rsi_14'] * X['macd_hist']
    # keep only last row for prediction (most recent)
    last = X.iloc[-1:].copy()
    return last

def align_and_scale(last_row_df, scaler, feat_list):
    if feat_list is None:
        # use numeric columns in last_row_df
        X = last_row_df.select_dtypes(include=[np.number]).values.reshape(1, -1)
        return scaler.transform(X), list(last_row_df.select_dtypes(include=[np.number]).columns)
    # build vector in order of feat_list
    vec = []
    for f in feat_list:
        if f in last_row_df.columns:
            vec.append(float(last_row_df[f].iloc[0]))
        else:
            vec.append(0.0)
    X = np.array(vec, dtype=float).reshape(1, -1)
    Xs = scaler.transform(X)
    return Xs, feat_list

def calc_lot_from_fixed_risk(stop_distance, fixed_risk, pip_value_per_lot=PIP_VALUE_PER_LOT, pip_unit=PIP):
    if stop_distance <= 0:
        return 0.0
    pip_dist = abs(stop_distance) / pip_unit
    dollar_risk_per_lot = pip_dist * pip_value_per_lot
    if dollar_risk_per_lot <= 0:
        return 0.0
    lots = fixed_risk / dollar_risk_per_lot
    return float(lots)

def clamp_lot(symbol_info, lots):
    min_lot = getattr(symbol_info, "volume_min", 0.01)
    max_lot = getattr(symbol_info, "volume_max", 100.0)
    step = getattr(symbol_info, "volume_step", 0.01)
    lots = max(lots, min_lot)
    lots = min(lots, max_lot)
    # round to nearest step
    steps = round((lots - min_lot) / step)
    lots = min_lot + steps * step
    # final clamp
    lots = max(min_lot, min(lots, max_lot))
    return round(lots, 2)

def send_market_order(symbol, lot, buy_sell, price, sl, tp, comment="auto_live"):
    type_ = mt5.ORDER_TYPE_BUY if buy_sell == "BUY" else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot),
        "type": type_,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 123456,
        "comment": comment,
        "type_time": ORDER_TIME,
        "type_filling": ORDER_FILLING,
    }
    res = mt5.order_send(request)
    return res

# ---------------- main loop ----------------
def main_loop():
    logging.info("Starting live autotrader main loop.")
    model, scaler, feat_list = load_model_and_scaler()
    connect_mt5()

    # initial equity log (account)
    account_info = mt5.account_info()
    if account_info is not None:
        equity = account_info.balance
    else:
        equity = None
    if equity is not None:
        pd.DataFrame([{"timestamp": human_time(), "balance": equity}]).to_csv(EQUITY_LOG, index=False, mode="a", header=not EQUITY_LOG.exists())

    try:
        while True:
            if STOP_FILE.exists():
                logging.warning("STOP file detected. Exiting main loop.")
                break

            # fetch candles and compute features
            try:
                raw = fetch_candles(N_BARS, TIMEFRAME)
                last_features = build_live_features(raw)  # one-row dataframe
            except Exception as e:
                logging.exception("Failed to fetch/build live features: %s", e)
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # align and scale
            try:
                Xs, used_features = align_and_scale(last_features, scaler, feat_list)
            except Exception as e:
                logging.exception("Scaling/align failed: %s", e)
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # predict
            try:
                prob = float(model.predict(Xs)[0])
                pred = 1 if prob > 0.55 else 0
                signal = "BUY" if pred == 1 else "SELL"
            except Exception as e:
                logging.exception("Model predict failed: %s", e)
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # check existing positions
            positions = mt5.positions_get(symbol=SYMBOL)
            if ONLY_ONE_POSITION_PER_SYMBOL and positions and len(positions) > 0:
                logging.info("Existing open positions for symbol %s found (%d). Skipping open.", SYMBOL, len(positions))
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # get live tick
            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is None:
                logging.warning("No tick available, skipping this cycle.")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            bid, ask = float(tick.bid), float(tick.ask)

            # choose entry: for BUY use ask, for SELL use bid; adjust for spread/slippage
            slippage = random.uniform(0.0, SLIPPAGE_MAX)
            if signal == "BUY":
                entry_price = ask + SPREAD / 2.0 + slippage
            else:
                entry_price = bid - SPREAD / 2.0 - slippage

            # compute stop distance using ATR from last_features
            atr_col = [c for c in last_features.columns if c.startswith("atr")]
            if atr_col:
                stop_distance = float(last_features[atr_col[0]].iloc[0])
            else:
                stop_distance = 0.0010
            stop_distance = max(stop_distance, 0.0010)

            if signal == "BUY":
                sl_price = entry_price - stop_distance
                tp_price = entry_price + REWARD_TO_RISK * stop_distance
            else:
                sl_price = entry_price + stop_distance
                tp_price = entry_price - REWARD_TO_RISK * stop_distance

            # calculate lot sizing
            symbol_info = mt5.symbol_info(SYMBOL)
            lots_to_send = 0.01
            if MODE == "fixed_dollars":
                lots_raw = calc_lot_from_fixed_risk(stop_distance, FIXED_RISK_DOLLARS, PIP_VALUE_PER_LOT, PIP)
                lots_to_send = clamp_lot(symbol_info, lots_raw) if symbol_info else round(lots_raw, 2)
            else:  # percent_of_balance
                # read account balance
                account_info = mt5.account_info()
                current_balance = float(account_info.balance) if account_info else 0.0
                risk_amount = current_balance * RISK_PERCENT
                lots_raw = calc_lot_from_fixed_risk(stop_distance, risk_amount, PIP_VALUE_PER_LOT, PIP)
                lots_to_send = clamp_lot(symbol_info, lots_raw) if symbol_info else round(lots_raw, 2)

            # safety check: minimal lot and non-zero
            if lots_to_send < 0.01:
                logging.warning("Calculated lots < broker min (0.01). Skipping trade. lots=%s", lots_to_send)
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # present summary in logs (this is auto mode; no human prompt)
            logging.info("Signal=%s prob=%.4f entry=%.5f SL=%.5f TP=%.5f lots=%.2f", signal, prob, entry_price, sl_price, tp_price, lots_to_send)

            # send order
            res = send_market_order(SYMBOL, lots_to_send, signal, entry_price, sl_price, tp_price, comment="auto_live")
            logging.info("Order result: %s", res)
            # log to CSV
            log_row = {
                "timestamp": human_time(),
                "symbol": SYMBOL,
                "signal": signal,
                "model_prob": prob,
                "entry_price": entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "lots": lots_to_send,
                "retcode": getattr(res, "retcode", None),
                "comment": getattr(res, "comment", None)
            }
            Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
            df_log = pd.DataFrame([log_row])
            if LOG_PATH.exists():
                df_log.to_csv(LOG_PATH, mode="a", header=False, index=False)
            else:
                df_log.to_csv(LOG_PATH, index=False)

            # small pause after trade
            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received, shutting down.")
    finally:
        disconnect_mt5()
        logging.info("Main loop ended. Exiting.")

if __name__ == "__main__":
    main_loop()