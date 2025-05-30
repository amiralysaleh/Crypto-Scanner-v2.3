# -*- coding: utf-8 -*-

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
import argparse
import os
import sys

# --- IMPORTS ---
try:
    from config import *
    # Import the same data preparation function used by the live analyzer
    from crypto_analyzer import prepare_dataframe
    # Import the live signal generator to be used in the backtest
    from signal_generator import generate_signals
except ImportError as e:
    print(f"Fatal Error: Cannot import project modules: {e}")
    sys.exit(1)

def fetch_historical_kline(symbol, start_dt, end_dt, interval):
    url = f"{KUCOIN_BASE_URL}{KUCOIN_KLINE_ENDPOINT}"
    all_data = []
    
    interval_map = {'15min': 900, '1hour': 3600, '4hour': 14400}
    interval_seconds = interval_map.get(interval, 900)
    
    current_end_ts = int(end_dt.timestamp())
    start_ts = int(start_dt.timestamp())

    print(f"Fetching data for {symbol} from {start_dt.date()} to {end_dt.date()} ({interval})...")
    while current_end_ts > start_ts:
        current_start_ts = max(start_ts, current_end_ts - 1500 * interval_seconds)
        params = {"symbol": symbol, "type": interval, "startAt": current_start_ts, "endAt": current_end_ts}
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json().get('data', [])
            if not data: break
            
            df_chunk = pd.DataFrame(data, columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"])
            all_data.append(df_chunk)
            oldest_ts_received = int(df_chunk.iloc[-1]["timestamp"])
            current_end_ts = oldest_ts_received - 1
            print(f"  ...fetched {len(df_chunk)} candles up to {datetime.fromtimestamp(oldest_ts_received)}")
            time.sleep(1.5)
        except Exception as e:
            print(f"  Error fetching data chunk: {e}. Retrying...")
            time.sleep(5)

    if not all_data: return None
    
    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df[["timestamp", "open", "close", "high", "low", "volume"]].astype(float)
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"], unit="s")
    full_df = full_df.sort_values("timestamp").reset_index(drop=True).drop_duplicates(subset=["timestamp"], keep='first')
    print(f"Fetched a total of {len(full_df)} unique candles for {interval}.")
    return full_df

def run_backtest(symbol, start_date_str, end_date_str):
    utc_tz = pytz.utc
    try:
        start_dt = utc_tz.localize(datetime.strptime(start_date_str, "%Y-%m-%d"))
        end_dt = utc_tz.localize(datetime.strptime(end_date_str, "%Y-%m-%d"))
    except ValueError:
        print("Error: Date format must be YYYY-MM-DD.")
        return

    print("\n--- Starting Backtest with High Win-Rate Strategy ---")
    print(f"Symbol: {symbol}, Period: {start_date_str} to {end_date_str}")
    print(f"Initial Capital: ${BACKTEST_SETTINGS['initial_capital']:,.2f}")
    print("---------------------------------------------------\n")

    # Fetch data for all three timeframes
    df_primary_raw = fetch_historical_kline(symbol, start_dt, end_dt, PRIMARY_TIMEFRAME)
    df_higher_raw = fetch_historical_kline(symbol, start_dt, end_dt, HIGHER_TIMEFRAME)
    df_trend_raw = fetch_historical_kline(symbol, start_dt, end_dt, TREND_TIMEFRAME)

    if any(df is None or df.empty for df in [df_primary_raw, df_higher_raw, df_trend_raw]):
        print("Failed to fetch sufficient historical data for one or more timeframes.")
        return

    capital = BACKTEST_SETTINGS['initial_capital']
    position = None
    trades = []
    equity_curve = [{'timestamp': df_primary_raw['timestamp'].iloc[0], 'capital': capital}]

    print(f"\nStarting simulation with {len(df_primary_raw)} primary candles...")
    # Loop through each candle of the primary timeframe
    for i in range(KLINE_SIZE, len(df_primary_raw)):
        # --- DATA PREPARATION FOR EACH STEP ---
        # Create rolling dataframes for each timeframe
        current_time = df_primary_raw['timestamp'].iloc[i]
        
        df_p_step = df_primary_raw.iloc[i-KLINE_SIZE:i+1].copy()
        df_h_step = df_higher_raw[df_higher_raw['timestamp'] <= current_time].iloc[-KLINE_SIZE:].copy()
        df_t_step = df_trend_raw[df_trend_raw['timestamp'] <= current_time].iloc[-KLINE_SIZE:].copy()

        if len(df_p_step) < 50 or len(df_h_step) < 50 or len(df_t_step) < 50: continue

        # Prepare dataframes with indicators for the current step
        prep_p = prepare_dataframe(df_p_step, PRIMARY_TIMEFRAME)
        prep_h = prepare_dataframe(df_h_step, HIGHER_TIMEFRAME)
        prep_t = prepare_dataframe(df_t_step, TREND_TIMEFRAME)

        if any(df is None or df.empty for df in [prep_p, prep_h, prep_t]): continue
        
        latest_candle = prep_p.iloc[-1]
        
        # --- POSITION MANAGEMENT ---
        if position:
            pnl = 0
            close_price = 0
            status = ""
            if position['type'] == 'BUY':
                if latest_candle['high'] >= position['tp']:
                    status, close_price = 'Target Reached', position['tp']
                elif latest_candle['low'] <= position['sl']:
                    status, close_price = 'Stop Loss Hit', position['sl']
            elif position['type'] == 'SELL':
                if latest_candle['low'] <= position['tp']:
                    status, close_price = 'Target Reached', position['tp']
                elif latest_candle['high'] >= position['sl']:
                    status, close_price = 'Stop Loss Hit', position['sl']

            if status:
                pnl = (close_price - position['entry']) * position['size'] if position['type'] == 'BUY' else (position['entry'] - close_price) * position['size']
                pnl -= position['commission'] # Subtract commission
                capital += pnl
                trades.append({
                    'Entry Time': position['entry_time'], 'Close Time': current_time,
                    'Type': position['type'], 'Status': status, 'Entry Price': position['entry'],
                    'Close Price': close_price, 'PNL': pnl, 'Score': position['score']
                })
                equity_curve.append({'timestamp': current_time, 'capital': capital})
                position = None
        
        # --- SIGNAL GENERATION ---
        if not position:
            # Use the actual signal generator
            signals = generate_signals(prep_p, prep_h, prep_t, symbol)
            if signals:
                signal = signals[0] # Take the first signal if multiple are generated
                position_size_usd = capital * (BACKTEST_SETTINGS['position_size_percent'] / 100)
                coin_size = position_size_usd / float(signal['entry_price'])
                commission_cost = position_size_usd * (BACKTEST_SETTINGS['commission_percent'] / 100) * 2 # Entry and Exit

                position = {
                    'type': signal['type'],
                    'entry': float(signal['entry_price']),
                    'tp': float(signal['target_price']),
                    'sl': float(signal['stop_loss']),
                    'size': coin_size,
                    'entry_time': current_time,
                    'score': signal['score'],
                    'commission': commission_cost
                }
                print(f"[{current_time.date()}] OPEN {signal['type']} at {signal['entry_price']:.4f} (Score: {signal['score']})")


    # --- RESULTS ANALYSIS ---
    print("\n--- Backtest Results ---")
    if not trades:
        print("No trades were executed.")
        return

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    wins = trades_df[trades_df['PNL'] > 0]
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = trades_df['PNL'].sum()
    profit_factor = abs(wins['PNL'].sum() / trades_df[trades_df['PNL'] < 0]['PNL'].sum()) if trades_df[trades_df['PNL'] < 0]['PNL'].sum() != 0 else float('inf')
    
    equity_df = pd.DataFrame(equity_curve)
    equity_df['peak'] = equity_df['capital'].cummax()
    equity_df['drawdown'] = equity_df['peak'] - equity_df['capital']
    max_drawdown = equity_df['drawdown'].max()
    max_drawdown_pct = (max_drawdown / equity_df.loc[equity_df['drawdown'].idxmax()]['peak']) * 100 if max_drawdown > 0 else 0


    print(f"Period:                {start_date_str} to {end_date_str}")
    print(f"Final Capital:         ${capital:,.2f}")
    print(f"Total PNL:             ${total_pnl:,.2f} ({(total_pnl/BACKTEST_SETTINGS['initial_capital'])*100:.2f}%)")
    print(f"Total Trades:          {total_trades}")
    print(f"Win Rate:              {win_rate:.2f}%")
    print(f"Profit Factor:         {profit_factor:.2f}")
    print(f"Max Drawdown:          ${max_drawdown:,.2f} ({max_drawdown_pct:.2f}%)")
    print("--------------------------\n")

    os.makedirs('data', exist_ok=True)
    result_filename = f"data/backtest_results_{symbol}_{start_date_str}_to_{end_date_str}.csv"
    trades_df.to_csv(result_filename, index=False)
    print(f"Results saved to {result_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a backtest for the crypto trading strategy.')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., BTC-USDT)')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    # Capital and Size are now read from config.py
    args = parser.parse_args()
    run_backtest(args.symbol, args.start, args.end)
