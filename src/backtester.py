# -*- coding: utf-8 -*-

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
import argparse
import traceback
import os  # <<<--- اطمینان از وجود import os
import sys

# --- CHANGED IMPORTS ---
# Use simple imports, relying on PYTHONPATH to find them in 'src'
try:
    from config import *
    from crypto_analyzer import prepare_dataframe
    # Note: We are NOT importing generate_signals anymore
except ImportError as e:
    print(f"Fatal Error: Cannot import project modules: {e}")
    print("Ensure 'src' is in PYTHONPATH or this script is run correctly.")
    sys.exit(1)
# --- END CHANGED IMPORTS ---

# --- Data Fetching (Copied here for simplicity) ---
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
            time.sleep(1.5)

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
    """
    current_price = latest['close']
    atr = latest['atr']
    ichi_base_period = SCALPING_SETTINGS['ichi_base_period']
    price_26_ago_index = latest.name - ichi_base_period
    if price_26_ago_index < 0: return None
    price_26_ago = df_primary['close'].iloc[price_26_ago_index]

    if latest['adx'] < SCALPING_SETTINGS['adx_threshold']: return None

    buy_factors = set()
    if latest['rsi'] < SCALPING_SETTINGS['rsi_oversold'] + 5: buy_factors.add('rsi')
    if latest['stoch_k'] < SCALPING_SETTINGS['stoch_oversold'] and latest['stoch_k'] > latest['stoch_d']: buy_factors.add('stoch')
    if prev['ema_short'] <= prev['ema_medium'] and latest['ema_short'] > latest['ema_medium']: buy_factors.add('ema')
    if prev['macd'] <= prev['macd_signal'] and latest['macd'] > latest['macd_signal']: buy_factors.add('macd')
    if latest['close'] <= latest['bb_lower']: buy_factors.add('bb')
    if (latest['close'] > latest['ichi_a'] and latest['close'] > latest['ichi_b'] and
        latest['ichi_conv'] > latest['ichi_base'] and latest['close'] > price_26_ago): buy_factors.add('ichi')
    if latest['candle_pattern'] == 'bullish_engulfing' or latest['candle_pattern'] == 'hammer': buy_factors.add('candle')
    if latest['rsi_divergence'] == 'bullish' or latest['macd_divergence'] == 'bullish': buy_factors.add('divergence')
    if higher_tf_trend == 'up': buy_factors.add('higher_tf')
    if latest['adx_pos'] > latest['adx_neg']: buy_factors.add('adx')

    if len(buy_factors) >= 3:
        target_price = current_price + (atr * SCALPING_SETTINGS['profit_target_multiplier'])
        stop_loss = current_price - (atr * SCALPING_SETTINGS['stop_loss_multiplier'])
        if (current_price - stop_loss) > 0:
            risk_reward_ratio = (target_price - current_price) / (current_price - stop_loss)
            if risk_reward_ratio >= SCALPING_SETTINGS['min_risk_reward_ratio']:
                return {'type': 'BUY', 'price': current_price, 'tp': target_price, 'sl': stop_loss}

    sell_factors = set()
    if latest['rsi'] > SCALPING_SETTINGS['rsi_overbought'] - 5: sell_factors.add('rsi')
    if latest['stoch_k'] > SCALPING_SETTINGS['stoch_oversold'] and latest['stoch_k'] < latest['stoch_d']: sell_factors.add('stoch')
    if prev['ema_short'] >= prev['ema_medium'] and latest['ema_short'] < latest['ema_medium']: sell_factors.add('ema')
    if prev['macd'] >= prev['macd_signal'] and latest['macd'] < latest['macd_signal']: sell_factors.add('macd')
    if latest['close'] >= latest['bb_upper']: sell_factors.add('bb')
    if (latest['close'] < latest['ichi_a'] and latest['close'] < latest['ichi_b'] and
        latest['ichi_conv'] < latest['ichi_base'] and latest['close'] < price_26_ago): sell_factors.add('ichi')
    if latest['candle_pattern'] == 'bearish_engulfing' or latest['candle_pattern'] == 'shooting_star': sell_factors.add('candle')
    if latest['rsi_divergence'] == 'bearish' or latest['macd_divergence'] == 'bearish': sell_factors.add('divergence')
    if higher_tf_trend == 'down': sell_factors.add('higher_tf')
    if latest['adx_neg'] > latest['adx_pos']: sell_factors.add('adx')

    if len(sell_factors) >= 3:
        target_price = current_price - (atr * SCALPING_SETTINGS['profit_target_multiplier'])
        stop_loss = current_price + (atr * SCALPING_SETTINGS['stop_loss_multiplier'])
        if (stop_loss - current_price) > 0:
            risk_reward_ratio = (current_price - target_price) / (stop_loss - current_price)
            if risk_reward_ratio >= SCALPING_SETTINGS['min_risk_reward_ratio']:
                return {'type': 'SELL', 'price': current_price, 'tp': target_price, 'sl': stop_loss}

    return None

def run_backtest(symbol, start_date_str, end_date_str, initial_capital, trade_size):
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
    print("-------------------------\n")

    df_primary_raw = fetch_historical_kline(symbol, start_dt, end_dt, PRIMARY_TIMEFRAME)
    df_higher_raw = fetch_historical_kline(symbol, start_dt, end_dt, HIGHER_TIMEFRAME)

    if df_primary_raw is None or df_higher_raw is None: return

    print("Preparing primary timeframe data...")
    df_primary = prepare_dataframe(df_primary_raw, PRIMARY_TIMEFRAME)
    print("Preparing higher timeframe data...")
    df_higher = prepare_dataframe(df_higher_raw, HIGHER_TIMEFRAME)

    if df_primary is None or df_higher is None or df_primary.empty or df_higher.empty: return

    df_higher = df_higher.set_index('timestamp')
    df_primary = df_primary.set_index('timestamp')
    df_higher_aligned = df_higher.reindex(df_primary.index, method='ffill').reset_index()
    df_primary = df_primary.reset_index()

    capital = initial_capital
    position, entry_price, target_price, stop_loss, entry_time = None, 0, 0, 0, None
    trades, equity_curve = [], [initial_capital]

    print(f"\nStarting simulation with {len(df_primary)} prepared candles...")
    start_index_loop = SCALPING_SETTINGS['ichi_base_period'] + 5

    for i in range(start_index_loop, len(df_primary)):
        latest = df_primary.iloc[i]
        prev = df_primary.iloc[i-1]
        higher_tf_trend = df_higher_aligned.iloc[i]['trend_confirmed']

        if position:
            hit, pnl, close_price, status = False, 0, 0, ""
            if position == 'BUY' and latest['high'] >= target_price: close_price, status, hit = target_price, 'Target Reached', True
            elif position == 'BUY' and latest['low'] <= stop_loss: close_price, status, hit = stop_loss, 'Stop Loss Hit', True
            elif position == 'SELL' and latest['low'] <= target_price: close_price, status, hit = target_price, 'Target Reached', True
            elif position == 'SELL' and latest['high'] >= stop_loss: close_price, status, hit = stop_loss, 'Stop Loss Hit', True

            if hit:
                pnl = (close_price - entry_price) * (trade_size / entry_price) if position == 'BUY' else (entry_price - close_price) * (trade_size / entry_price)
                capital += pnl
                trades.append({'Symbol': symbol, 'Type': position, 'Status': status, 'Entry Time': entry_time, 'Close Time': latest['timestamp'], 'Entry Price': entry_price, 'Close Price': close_price, 'PNL': pnl, 'Capital': capital})
                position = None
                equity_curve.append(capital)

        if not position:
            signal = apply_signal_logic(latest, prev, higher_tf_trend, df_primary, symbol)
            if signal:
                position, entry_price, target_price, stop_loss, entry_time = signal['type'], signal['price'], signal['tp'], signal['sl'], latest['timestamp']
                print(f"[{latest['timestamp']}] OPENED {position} at {entry_price:.4f}. TP: {target_price:.4f}, SL: {stop_loss:.4f}")

    print("\n--- Backtest Results ---")
    if not trades: print("No trades were executed."); return

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df); wins = trades_df[trades_df['PNL'] > 0]; losses = trades_df[trades_df['PNL'] <= 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    avg_win = wins['PNL'].mean() if not wins.empty else 0; avg_loss = losses['PNL'].mean() if not losses.empty else 0
    profit_factor = abs(wins['PNL'].sum() / losses['PNL'].sum()) if not losses.empty and losses['PNL'].sum() != 0 else float('inf')
    total_pnl = trades_df['PNL'].sum(); final_capital = capital

    equity_df = pd.DataFrame(equity_curve, columns=['Equity'])
    equity_df['Peak'] = equity_df['Equity'].cummax(); equity_df['Drawdown'] = equity_df['Peak'] - equity_df['Equity']
    equity_df['Drawdown_Pct'] = (equity_df['Drawdown'] / equity_df['Peak']) * 100
    max_drawdown = equity_df['Drawdown'].max(); max_drawdown_pct = equity_df['Drawdown_Pct'].max() if not equity_df['Drawdown_Pct'].empty else 0

    print(f"Total Trades: {total_trades}"); print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Win: ${avg_win:.2f}"); print(f"Average Loss: ${avg_loss:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}"); print(f"Total PNL: ${total_pnl:.2f} ({ (total_pnl/initial_capital)*100 :.2f}%)")
    print(f"Final Capital: ${final_capital:,.2f}"); print(f"Max Drawdown: ${max_drawdown:,.2f} ({max_drawdown_pct:.2f}%)")
    print("------------------------\n")

    os.makedirs('data', exist_ok=True) # <<<--- os.makedirs now works
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
