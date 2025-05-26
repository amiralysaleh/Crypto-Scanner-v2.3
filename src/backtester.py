# -*- coding: utf-8 -*-

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
import argparse
import traceback
import os
import sys

# Add the project root to sys.path to handle imports correctly
# This assumes backtester.py is in 'src' and we want to import from 'src'
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(project_root)

try:
    from src.config import *
    from src.crypto_analyzer import prepare_dataframe
    # We won't use generate_signals directly, but keep config
except ImportError:
    print("Error: Could not import project modules. Ensure src/__init__.py exists and PYTHONPATH is set or run with 'python -m src.backtester'")
    # Fallback for simpler structures if run from root
    from config import *
    from crypto_analyzer import prepare_dataframe


# --- Data Fetching (Adapted for Backtesting) ---

def fetch_historical_kline(symbol, start_dt, end_dt, interval="30min"):
    """Fetch historical kline data from KuCoin in chunks."""
    url = f"{KUCOIN_BASE_URL}{KUCOIN_KLINE_ENDPOINT}"
    all_data = []

    if 'min' in interval:
        interval_seconds = int(interval.replace('min', '')) * 60
    elif 'hour' in interval:
        interval_seconds = int(interval.replace('hour', '')) * 3600
    else:
        interval_seconds = 1800 # Default

    current_end_ts = int(end_dt.timestamp())
    start_ts = int(start_dt.timestamp())

    print(f"Fetching data for {symbol} from {start_dt} to {end_dt} ({interval})...")

    while current_end_ts > start_ts:
        current_start_ts = max(start_ts, current_end_ts - 1500 * interval_seconds)
        params = {"symbol": symbol, "type": interval, "startAt": current_start_ts, "endAt": current_end_ts}

        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json().get('data', [])
            if not data:
                print(f"No more data received or error. End timestamp: {datetime.fromtimestamp(current_end_ts)}. Breaking.")
                break

            df_chunk = pd.DataFrame(data, columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"])
            all_data.append(df_chunk)
            oldest_ts_received = int(df_chunk.iloc[-1]["timestamp"])
            current_end_ts = oldest_ts_received - 1
            print(f"Fetched {len(df_chunk)} candles up to {datetime.fromtimestamp(oldest_ts_received)}")
            time.sleep(1.5) # Be nice to the API

        except Exception as e:
            print(f"Error fetching data chunk: {e}. Retrying in 5s...")
            time.sleep(5)

    if not all_data:
        print("No data fetched.")
        return None

    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df[["timestamp", "open", "close", "high", "low", "volume"]]
    full_df = full_df.astype(float)
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], unit="s")
    full_df = full_df.sort_values("timestamp").reset_index(drop=True)
    full_df = full_df.drop_duplicates(subset=["timestamp"], keep='first')
    print(f"Fetched a total of {len(full_df)} unique candles.")
    return full_df

# --- Backtester Core ---

def apply_signal_logic(latest, prev, higher_tf_trend, df_primary, symbol):
    """
    Applies the signal generation logic based on 'latest' and 'prev' rows.
    Returns a signal dictionary or None.
    This function *mimics* signal_generator.py logic.
    """
    current_price = latest['close']
    atr = latest['atr']
    ichi_base_period = SCALPING_SETTINGS['ichi_base_period']
    price_26_ago = df_primary['close'].iloc[latest.name - ichi_base_period]


    if latest['adx'] < SCALPING_SETTINGS['adx_threshold']:
        return None # Skip if ADX is low

    buy_factors = set()
    buy_reasons = []

    # RSI
    if latest['rsi'] < SCALPING_SETTINGS['rsi_oversold'] + 5:
        buy_factors.add('rsi')
        buy_reasons.append(f"RSI near oversold ({latest['rsi']:.2f})")
    # Stochastic
    if latest['stoch_k'] < SCALPING_SETTINGS['stoch_oversold'] and latest['stoch_k'] > latest['stoch_d']:
        buy_factors.add('stoch')
        buy_reasons.append(f"Stoch K > D in oversold")
    # EMA Cross
    if prev['ema_short'] <= prev['ema_medium'] and latest['ema_short'] > latest['ema_medium']:
        buy_factors.add('ema')
        buy_reasons.append("EMA Short/Medium bullish cross")
    # MACD Cross
    if prev['macd'] <= prev['macd_signal'] and latest['macd'] > latest['macd_signal']:
        buy_factors.add('macd')
        buy_reasons.append("MACD bullish cross")
    # BB
    if latest['close'] <= latest['bb_lower']:
        buy_factors.add('bb')
        buy_reasons.append("Price at/below lower BB")
    # Ichimoku
    if (latest['close'] > latest['ichi_a'] and latest['close'] > latest['ichi_b'] and
        latest['ichi_conv'] > latest['ichi_base'] and
        latest['close'] > price_26_ago):
        buy_factors.add('ichi')
        buy_reasons.append("Strong Ichimoku bullish")
    # Candle
    if latest['candle_pattern'] == 'bullish_engulfing' or latest['candle_pattern'] == 'hammer':
        buy_factors.add('candle')
        buy_reasons.append(f"Bullish candle ({latest['candle_pattern']})")
    # Divergence
    if latest['rsi_divergence'] == 'bullish' or latest['macd_divergence'] == 'bullish':
         buy_factors.add('divergence')
         buy_reasons.append("Bullish Divergence")
    # Higher TF
    if higher_tf_trend == 'up':
        buy_factors.add('higher_tf')
        buy_reasons.append("Higher TF Up")
    # ADX
    if latest['adx_pos'] > latest['adx_neg']:
        buy_factors.add('adx')
        buy_reasons.append("ADX Bullish")

    if len(buy_reasons) >= 3:
        score = 65 # Simplification: Assume minimum score is met if enough reasons
        target_price = current_price + (atr * SCALPING_SETTINGS['profit_target_multiplier'])
        stop_loss = current_price - (atr * SCALPING_SETTINGS['stop_loss_multiplier'])
        if (current_price - stop_loss) > 0:
            risk_reward_ratio = (target_price - current_price) / (current_price - stop_loss)
            if risk_reward_ratio >= SCALPING_SETTINGS['min_risk_reward_ratio']:
                return {'type': 'BUY', 'price': current_price, 'tp': target_price, 'sl': stop_loss}

    # --- Sell Logic ---
    sell_factors = set()
    sell_reasons = []
    # RSI
    if latest['rsi'] > SCALPING_SETTINGS['rsi_overbought'] - 5:
        sell_factors.add('rsi')
        sell_reasons.append(f"RSI near overbought")
    # Stoch
    if latest['stoch_k'] > SCALPING_SETTINGS['stoch_overbought'] and latest['stoch_k'] < latest['stoch_d']:
        sell_factors.add('stoch')
        sell_reasons.append(f"Stoch K < D in overbought")
    # EMA
    if prev['ema_short'] >= prev['ema_medium'] and latest['ema_short'] < latest['ema_medium']:
        sell_factors.add('ema')
        sell_reasons.append("EMA Bearish cross")
    # MACD
    if prev['macd'] >= prev['macd_signal'] and latest['macd'] < latest['macd_signal']:
        sell_factors.add('macd')
        sell_reasons.append("MACD bearish cross")
    # BB
    if latest['close'] >= latest['bb_upper']:
        sell_factors.add('bb')
        sell_reasons.append("Price at/above upper BB")
    # Ichimoku
    if (latest['close'] < latest['ichi_a'] and latest['close'] < latest['ichi_b'] and
        latest['ichi_conv'] < latest['ichi_base'] and
        latest['close'] < price_26_ago):
        sell_factors.add('ichi')
        sell_reasons.append("Strong Ichimoku bearish")
    # Candle
    if latest['candle_pattern'] == 'bearish_engulfing' or latest['candle_pattern'] == 'shooting_star':
        sell_factors.add('candle')
        sell_reasons.append(f"Bearish candle ({latest['candle_pattern']})")
    # Divergence
    if latest['rsi_divergence'] == 'bearish' or latest['macd_divergence'] == 'bearish':
         sell_factors.add('divergence')
         sell_reasons.append("Bearish Divergence")
    # Higher TF
    if higher_tf_trend == 'down':
        sell_factors.add('higher_tf')
        sell_reasons.append("Higher TF Down")
    # ADX
    if latest['adx_neg'] > latest['adx_pos']:
        sell_factors.add('adx')
        sell_reasons.append("ADX Bearish")

    if len(sell_reasons) >= 3:
        score = 65 # Simplification
        target_price = current_price - (atr * SCALPING_SETTINGS['profit_target_multiplier'])
        stop_loss = current_price + (atr * SCALPING_SETTINGS['stop_loss_multiplier'])
        if (stop_loss - current_price) > 0:
            risk_reward_ratio = (current_price - target_price) / (stop_loss - current_price)
            if risk_reward_ratio >= SCALPING_SETTINGS['min_risk_reward_ratio']:
                return {'type': 'SELL', 'price': current_price, 'tp': target_price, 'sl': stop_loss}

    return None # No signal


def run_backtest(symbol, start_date_str, end_date_str, initial_capital, trade_size):
    """Runs the backtest simulation."""
    tehran_tz = pytz.timezone('Asia/Tehran')
    try:
        start_dt = tehran_tz.localize(datetime.strptime(start_date_str, "%Y-%m-%d"))
        end_dt = tehran_tz.localize(datetime.strptime(end_date_str, "%Y-%m-%d"))
    except ValueError:
        print("Error: Date format must be YYYY-MM-DD.")
        return

    print("\n--- Starting Backtest ---")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date_str} to {end_date_str}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Trade Size: ${trade_size:,.2f}")
    print("-------------------------\n")

    df_primary_raw = fetch_historical_kline(symbol, start_dt, end_dt, PRIMARY_TIMEFRAME)
    df_higher_raw = fetch_historical_kline(symbol, start_dt, end_dt, HIGHER_TIMEFRAME)

    if df_primary_raw is None or df_higher_raw is None:
        print("Failed to fetch data.")
        return

    print("Preparing primary timeframe data...")
    df_primary = prepare_dataframe(df_primary_raw, PRIMARY_TIMEFRAME)
    print("Preparing higher timeframe data...")
    df_higher = prepare_dataframe(df_higher_raw, HIGHER_TIMEFRAME)

    if df_primary is None or df_higher is None or df_primary.empty or df_higher.empty:
        print("Failed to prepare dataframes.")
        return

    df_higher = df_higher.set_index('timestamp')
    df_primary = df_primary.set_index('timestamp')
    df_higher_aligned = df_higher.reindex(df_primary.index, method='ffill').reset_index()
    df_primary = df_primary.reset_index()

    capital = initial_capital
    position = None
    entry_price = 0
    target_price = 0
    stop_loss = 0
    entry_time = None
    trades = []
    equity_curve = [initial_capital]

    print(f"\nStarting simulation with {len(df_primary)} prepared candles...")

    start_index_loop = SCALPING_SETTINGS['ichi_base_period'] + 5 # Ensure enough history for Chikou

    for i in range(start_index_loop, len(df_primary)):
        latest = df_primary.iloc[i]
        prev = df_primary.iloc[i-1]
        higher_tf_row = df_higher_aligned.iloc[i]
        higher_tf_trend = higher_tf_row['trend_confirmed']

        # --- Check for closing positions ---
        if position:
            hit = False
            pnl = 0
            close_price = 0
            status = ""
            if position == 'BUY':
                if latest['high'] >= target_price:
                    close_price, status, hit = target_price, 'Target Reached', True
                elif latest['low'] <= stop_loss:
                    close_price, status, hit = stop_loss, 'Stop Loss Hit', True
            elif position == 'SELL':
                if latest['low'] <= target_price:
                    close_price, status, hit = target_price, 'Target Reached', True
                elif latest['high'] >= stop_loss:
                    close_price, status, hit = stop_loss, 'Stop Loss Hit', True

            if hit:
                pnl = (close_price - entry_price) * (trade_size / entry_price) if position == 'BUY' else (entry_price - close_price) * (trade_size / entry_price)
                capital += pnl
                trades.append({
                    'Symbol': symbol, 'Type': position, 'Status': status,
                    'Entry Time': entry_time, 'Close Time': latest['timestamp'],
                    'Entry Price': entry_price, 'Close Price': close_price,
                    'Target': target_price, 'Stop Loss': stop_loss,
                    'PNL': pnl, 'Capital': capital
                })
                print(f"[{latest['timestamp']}] CLOSED {position} at {close_price:.4f}. PNL: ${pnl:.2f}. Capital: ${capital:.2f}")
                position = None
                equity_curve.append(capital)

        # --- Check for opening new positions ---
        if not position:
            signal = apply_signal_logic(latest, prev, higher_tf_trend, df_primary, symbol)
            if signal:
                position = signal['type']
                entry_price = signal['price']
                target_price = signal['tp']
                stop_loss = signal['sl']
                entry_time = latest['timestamp']
                print(f"[{latest['timestamp']}] OPENED {position} at {entry_price:.4f}. TP: {target_price:.4f}, SL: {stop_loss:.4f}")

    print("\n--- Backtest Results ---")
    if not trades:
        print("No trades were executed.")
        return

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    wins = trades_df[trades_df['PNL'] > 0]
    losses = trades_df[trades_df['PNL'] <= 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    avg_win = wins['PNL'].mean() if not wins.empty else 0
    avg_loss = losses['PNL'].mean() if not losses.empty else 0
    profit_factor = abs(wins['PNL'].sum() / losses['PNL'].sum()) if not losses.empty and losses['PNL'].sum() != 0 else float('inf')
    total_pnl = trades_df['PNL'].sum()
    final_capital = capital

    equity_df = pd.DataFrame(equity_curve, columns=['Equity'])
    equity_df['Peak'] = equity_df['Equity'].cummax()
    equity_df['Drawdown'] = equity_df['Peak'] - equity_df['Equity']
    equity_df['Drawdown_Pct'] = (equity_df['Drawdown'] / equity_df['Peak']) * 100
    max_drawdown = equity_df['Drawdown'].max()
    max_drawdown_pct = equity_df['Drawdown_Pct'].max()

    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Win: ${avg_win:.2f}")
    print(f"Average Loss: ${avg_loss:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Total PNL: ${total_pnl:.2f} ({ (total_pnl/initial_capital)*100 :.2f}%)")
    print(f"Final Capital: ${final_capital:,.2f}")
    print(f"Max Drawdown: ${max_drawdown:,.2f} ({max_drawdown_pct:.2f}%)")
    print("------------------------\n")

    # Ensure data directory exists before saving
    os.makedirs('data', exist_ok=True)
    result_filename = f"data/backtest_results_{symbol}_{start_date_str}_to_{end_date_str}.csv"
    trades_df.to_csv(result_filename, index=False)
    print(f"Results saved to {result_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a backtest for the crypto trading strategy.')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., BTC-USDT)')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--size', type=float, default=1000, help='Fixed trade size in USDT')
    args = parser.parse_args()
    run_backtest(args.symbol, args.start, args.end, args.capital, args.size)
