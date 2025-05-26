# -*- coding: utf-8 -*-

import requests
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import pytz
import ta
import traceback
from scipy.signal import argrelextrema

from config import *
from signal_generator import generate_signals
from telegram_sender import send_telegram_message
from signal_tracker import save_signal, load_signals

# ... (تمام توابع دیگر بدون تغییر باقی می‌مانند) ...

def fetch_kline_data(symbol, size=100, interval="30min"):
    """Fetch kline data from KuCoin with retry"""
    url = f"{KUCOIN_BASE_URL}{KUCOIN_KLINE_ENDPOINT}"
    end_time = int(time.time())

    # Calculate interval in seconds (handle different units)
    if 'min' in interval:
        interval_seconds = int(interval.replace('min', '')) * 60
    elif 'hour' in interval:
        interval_seconds = int(interval.replace('hour', '')) * 3600
    else:
        interval_seconds = 1800 # Default to 30min

    start_time = end_time - (size * interval_seconds)

    params = {"symbol": symbol, "type": interval, "startAt": start_time, "endAt": end_time}
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=15) # Increased timeout
            response.raise_for_status()
            data = response.json()
            if not data.get('data'):
                print(f"Error fetching data for {symbol} on {interval}: {data}")
                return None
            df = pd.DataFrame(data['data'], columns=[
                "timestamp", "open", "close", "high", "low", "volume", "turnover"
            ])
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df = df.astype(float)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df.iloc[::-1].reset_index(drop=True)
            print(f"Received {len(df)} candles for {symbol} on {interval}")
            return df
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {symbol} on {interval}: {e}")
            time.sleep(2 ** attempt)
    return None

def fetch_volume_data(symbol):
    """Fetch 24h trading volume from KuCoin"""
    url = f"{KUCOIN_BASE_URL}{KUCOIN_STATS_ENDPOINT}"
    params = {"symbol": symbol}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        volume = float(data.get('data', {}).get('volValue', 0))
        print(f"24h volume for {symbol}: {volume} USDT")
        return volume
    except Exception as e:
        print(f"Error fetching volume for {symbol}: {e}")
        return 0

def check_trend_consistency(trend_series):
    """Check trend consistency in the time window"""
    if len(trend_series) == 0:
        return 'neutral'
    if all(trend == 'up' for trend in trend_series):
        return 'up'
    if all(trend == 'down' for trend in trend_series):
        return 'down'
    return 'neutral'

def identify_candlestick_patterns(df):
    """Identify simple candlestick patterns."""
    patterns = []
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        pattern = 'none'

        # Bullish Engulfing
        if (curr['close'] > curr['open'] and prev['close'] < prev['open'] and
            curr['close'] > prev['open'] and curr['open'] < prev['close']):
            pattern = 'bullish_engulfing'
        # Bearish Engulfing
        elif (curr['close'] < curr['open'] and prev['close'] > prev['open'] and
              curr['open'] > prev['close'] and curr['close'] < prev['open']):
            pattern = 'bearish_engulfing'
        # Hammer (Simplified) - Needs context (downtrend)
        elif (curr['close'] > curr['open'] and
              (curr['high'] - curr['close']) < 0.2 * (curr['high'] - curr['low']) and
              (curr['open'] - curr['low']) > 2 * (curr['close'] - curr['open'])):
             pattern = 'hammer'
        # Shooting Star (Simplified) - Needs context (uptrend)
        elif (curr['open'] > curr['close'] and
              (curr['high'] - curr['open']) > 2 * (curr['open'] - curr['close']) and
              (curr['close'] - curr['low']) < 0.2 * (curr['high'] - curr['low'])):
             pattern = 'shooting_star'

        patterns.append(pattern)
    # Add 'none' for the first row as it has no previous candle
    return ['none'] + patterns

def find_divergence(price, indicator, lookback=14, order=5):
    """
    Finds simple divergence (Bullish/Bearish Regular)
    order: How many points on each side to use for local extrema.
    Returns: 'bullish', 'bearish', or 'none'.
    """
    if len(price) < lookback or len(indicator) < lookback:
        return 'none'

    price_subset = price.iloc[-lookback:]
    indicator_subset = indicator.iloc[-lookback:]

    # Find local minima (lows) and maxima (highs)
    low_indices = argrelextrema(price_subset.values, np.less, order=order)[0]
    high_indices = argrelextrema(price_subset.values, np.greater, order=order)[0]

    # Bullish Divergence (Lower Lows in Price, Higher Lows in Indicator)
    if len(low_indices) >= 2:
        last_low_idx = low_indices[-1]
        prev_low_idx = low_indices[-2]
        if (price_subset.iloc[last_low_idx] < price_subset.iloc[prev_low_idx] and
            indicator_subset.iloc[last_low_idx] > indicator_subset.iloc[prev_low_idx]):
            return 'bullish'

    # Bearish Divergence (Higher Highs in Price, Lower Highs in Indicator)
    if len(high_indices) >= 2:
        last_high_idx = high_indices[-1]
        prev_high_idx = high_indices[-2]
        if (price_subset.iloc[last_high_idx] > price_subset.iloc[prev_high_idx] and
            indicator_subset.iloc[last_high_idx] < indicator_subset.iloc[prev_high_idx]):
            return 'bearish'

    return 'none'


def prepare_dataframe(df, timeframe=PRIMARY_TIMEFRAME):
    """Add technical indicators and price action rules (Enhanced)"""
    if df is None or len(df) < SCALPING_SETTINGS['trend_confirmation_window'] or len(df) < SCALPING_SETTINGS['ichi_span_b_period']:
        print(f"Not enough data to prepare DataFrame for {timeframe} ({len(df)} candles)")
        return None
    try:
        # Standard Indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=SCALPING_SETTINGS['rsi_period']).rsi()
        df['ema_short'] = ta.trend.ema_indicator(df['close'], window=SCALPING_SETTINGS['ema_short'])
        df['ema_medium'] = ta.trend.ema_indicator(df['close'], window=SCALPING_SETTINGS['ema_medium'])
        df['ema_long'] = ta.trend.ema_indicator(df['close'], window=SCALPING_SETTINGS['ema_long'])

        macd = ta.trend.MACD(df['close'],
                           window_fast=SCALPING_SETTINGS['macd_fast'],
                           window_slow=SCALPING_SETTINGS['macd_slow'],
                           window_sign=SCALPING_SETTINGS['macd_signal'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        bollinger = ta.volatility.BollingerBands(df['close'],
                                              window=SCALPING_SETTINGS['bb_period'],
                                              window_dev=SCALPING_SETTINGS['bb_std'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()

        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=14
        ).average_true_range()

        # New Indicators
        stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'],
                                               window=SCALPING_SETTINGS['stoch_k'],
                                               smooth_window=SCALPING_SETTINGS['stoch_d'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        adx = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'],
                                  window=SCALPING_SETTINGS['adx_period'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos() # +DI
        df['adx_neg'] = adx.adx_neg() # -DI

        ichi = ta.trend.IchimokuIndicator(high=df['high'], low=df['low'],
                                        window1=SCALPING_SETTINGS['ichi_conv_period'],
                                        window2=SCALPING_SETTINGS['ichi_base_period'],
                                        window3=SCALPING_SETTINGS['ichi_span_b_period'],
                                        visual=True) # visual=True shifts A and B forward
        df['ichi_conv'] = ichi.ichimoku_conversion_line()
        df['ichi_base'] = ichi.ichimoku_base_line()
        df['ichi_a'] = ichi.ichimoku_a()
        df['ichi_b'] = ichi.ichimoku_b()

        # Price Action & Trend
        df['volume_change'] = df['volume'].pct_change()
        df['price_change'] = df['close'].pct_change()
        df['resistance'] = df['high'].rolling(window=10).max()
        df['support'] = df['low'].rolling(window=10).min()
        df['trend'] = np.where(df['ema_short'] > df['ema_long'], 'up', 'down')

        window = SCALPING_SETTINGS['trend_confirmation_window']
        trend_confirmed = []
        for i in range(len(df)):
            if i < window - 1:
                trend_confirmed.append('neutral')
            else:
                trend_slice = df['trend'].iloc[i - window + 1:i + 1]
                trend_confirmed.append(check_trend_consistency(trend_slice))
        df['trend_confirmed'] = trend_confirmed

        # Candlestick Patterns
        df['candle_pattern'] = identify_candlestick_patterns(df)

        # Divergence (Check only for the latest candle for performance)
        df['rsi_divergence'] = 'none'
        df['macd_divergence'] = 'none'
        if len(df) > SCALPING_SETTINGS['divergence_lookback']:
             df.loc[df.index[-1], 'rsi_divergence'] = find_divergence(df['close'], df['rsi'], SCALPING_SETTINGS['divergence_lookback'])
             df.loc[df.index[-1], 'macd_divergence'] = find_divergence(df['close'], df['macd_diff'], SCALPING_SETTINGS['divergence_lookback'])

        # Drop NaN values created by indicators
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
    except Exception as e:
        print(f"Error preparing DataFrame for {timeframe}: {e}")
        print(traceback.format_exc())
        return None

def generate_tradingview_link(symbol):
    """Generate TradingView chart link for the given symbol"""
    tradingview_symbol = symbol.replace('-', '')
    return f"https://www.tradingview.com/chart/?symbol=KUCOIN:{tradingview_symbol}"

def main():
    print("🚀 Starting cryptocurrency analysis (Enhanced)...")
    signals_sent = 0
    tehran_tz = pytz.timezone('Asia/Tehran')
    active_signals = {s['symbol']: s for s in load_signals() if s['status'] == 'active'}

    for crypto in CRYPTOCURRENCIES:
        print(f"\nAnalyzing {crypto}...")
        try:
            trading_symbol = KUCOIN_SUPPORTED_PAIRS.get(crypto, crypto)
            if trading_symbol != crypto:
                print(f"Using {trading_symbol} instead of {crypto}")

            volume_24h = fetch_volume_data(trading_symbol)
            if volume_24h < SCALPING_SETTINGS['min_volume_threshold']:
                print(f"Skipping {crypto} due to low 24h volume: {volume_24h}")
                continue

            if crypto in active_signals:
                try:
                    created_at_str = active_signals[crypto]['created_at']
                    # Handle both ISO format and older format robustly
                    if 'T' in created_at_str and ('+' in created_at_str or 'Z' in created_at_str):
                         created_at = datetime.fromisoformat(created_at_str)
                    else:
                         created_at = tehran_tz.localize(datetime.strptime(created_at_str, "%Y-%m-%d %H:%M:%S"))

                    time_diff = (datetime.now(tehran_tz) - created_at).total_seconds() / 60
                    if time_diff < SCALPING_SETTINGS['signal_cooldown_minutes']:
                        print(f"Skipping {crypto} due to active signal cooldown ({time_diff:.1f}m < {SCALPING_SETTINGS['signal_cooldown_minutes']}m)")
                        continue
                except Exception as e:
                     print(f"Error processing cooldown for {crypto}: {e}. Continuing analysis.")


            df_primary = fetch_kline_data(trading_symbol, size=KLINE_SIZE, interval=PRIMARY_TIMEFRAME)
            if df_primary is None:
                continue

            df_higher = fetch_kline_data(trading_symbol, size=KLINE_SIZE // 2, interval=HIGHER_TIMEFRAME)
            if df_higher is None:
                continue

            prepared_df_primary = prepare_dataframe(df_primary, PRIMARY_TIMEFRAME)
            prepared_df_higher = prepare_dataframe(df_higher, HIGHER_TIMEFRAME)
            if prepared_df_primary is None or len(prepared_df_primary) == 0 or \
               prepared_df_higher is None or len(prepared_df_higher) == 0:
                print(f"Skipping {crypto} due to insufficient prepared data.")
                continue

            signals = generate_signals(prepared_df_primary, prepared_df_higher, crypto)
            for signal in signals:
                tradingview_link = generate_tradingview_link(signal['symbol'])
                # *** THIS IS THE CHANGED PART ***
                # Using <code> for prices and <b> for titles for better HTML.
                # Added <a href> for the link.
                message = (
                    f"<b>🚨 Signal {signal['type']} for {signal['symbol']} 🚨</b>\n\n"
                    f"💰 <b>Current Price:</b> <code>{signal['current_price']}</code>\n"
                    f"🎯 <b>Target Price:</b> <code>{signal['target_price']}</code>\n"
                    f"🛑 <b>Stop Loss:</b> <code>{signal['stop_loss']}</code>\n"
                    f"🏆 <b>Score:</b> {signal['score']}/100\n"
                    f"📈 <b>Risk/Reward:</b> {signal['risk_reward_ratio']:.2f}\n\n"
                    f"<b>📝 Reasons:</b>\n{signal['reasons']}\n\n"
                    f"📊 <b>View Chart:</b> <a href=\"{tradingview_link}\">TradingView Link</a>\n"
                    f"⏱️ <b>Time (Tehran):</b> {signal['time']}"
                )
                # *** END OF CHANGED PART ***
                if send_telegram_message(message):
                    signals_sent += 1
                    save_signal(signal)
                    print(f"✅ Signal sent and saved for {crypto}: {signal['type']}")
                else:
                    print(f"❌ Failed to send signal for {crypto}")

        except Exception as e:
            print(f"Error during analysis of {crypto}: {e}")
            print(traceback.format_exc())

        time.sleep(1) # Slightly increased sleep time

    send_telegram_message(f"✅ Scan completed. {signals_sent} signals sent.", silent=True)
    print(f"\nAnalysis complete. {signals_sent} signals sent.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        send_telegram_message(f"❌ System error: {e}")

