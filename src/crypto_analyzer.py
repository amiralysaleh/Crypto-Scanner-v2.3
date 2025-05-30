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
import warnings

# --- IMPORTS ---
from config import *
from signal_generator import generate_signals
from telegram_sender import send_telegram_message
from signal_tracker import save_signal, load_signals

warnings.filterwarnings('ignore')

def fetch_kline_data(symbol, size=200, interval="15min"):
    """Fetch kline data from KuCoin with enhanced error handling"""
    url = f"{KUCOIN_BASE_URL}{KUCOIN_KLINE_ENDPOINT}"
    end_time = int(time.time())

    interval_map = {
        '1min': 60, '3min': 180, '5min': 300, '15min': 900,
        '30min': 1800, '1hour': 3600, '2hour': 7200, '4hour': 14400,
        '6hour': 21600, '8hour': 28800, '12hour': 43200, '1day': 86400
    }
    
    interval_seconds = interval_map.get(interval, 900)
    start_time = end_time - (size * interval_seconds)

    params = {"symbol": symbol, "type": interval, "startAt": start_time, "endAt": end_time}
    
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('data') or len(data['data']) < 50:
                print(f"-> Insufficient data for {symbol} on {interval}: {len(data.get('data', []))} candles")
                return None
                
            df = pd.DataFrame(data['data'], columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"])
            df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype(float)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df = df.iloc[::-1].reset_index(drop=True)
            
            print(f"-> Fetched {len(df)} candles for {symbol} on {interval}")
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"--> Attempt {attempt + 1} failed for {symbol} on {interval}: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    
    print(f"--> Failed to fetch data for {symbol} after 3 attempts.")
    return None

def fetch_market_stats(symbol):
    """Fetch comprehensive market statistics"""
    url = f"{KUCOIN_BASE_URL}{KUCOIN_STATS_ENDPOINT}"
    params = {"symbol": symbol}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json().get('data', {})
        stats = {
            'volume_24h': float(data.get('volValue', 0)),
            'change_rate_24h': float(data.get('changeRate', 0)) * 100,
        }
        return stats
    except Exception as e:
        print(f"--> Error fetching market stats for {symbol}: {e}")
        return None

def analyze_candlestick_patterns(df):
    patterns = []
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        pattern = 'none'

        body = abs(curr['close'] - curr['open'])
        range_size = curr['high'] - curr['low']
        if range_size == 0:
            patterns.append('none')
            continue

        # Bullish Engulfing
        if (curr['close'] > curr['open'] and prev['close'] < prev['open'] and
            curr['close'] > prev['open'] and curr['open'] < prev['close'] and body > range_size * 0.6):
            pattern = 'bullish_engulfing'
        # Bearish Engulfing
        elif (curr['close'] < curr['open'] and prev['close'] > prev['open'] and
              curr['open'] > prev['close'] and curr['close'] < prev['open'] and body > range_size * 0.6):
            pattern = 'bearish_engulfing'
        # Hammer
        elif (curr['close'] > curr['open'] and
              (min(curr['open'], curr['close']) - curr['low']) > 2 * body and
              (curr['high'] - max(curr['open'], curr['close'])) < body * 0.5):
             pattern = 'hammer'
        # Shooting Star
        elif (curr['close'] < curr['open'] and
              (curr['high'] - max(curr['open'], curr['close'])) > 2 * body and
              (min(curr['open'], curr['close']) - curr['low']) < body * 0.5):
             pattern = 'shooting_star'
        patterns.append(pattern)
    return ['none'] + patterns


def find_divergence(price, indicator, lookback, order=5):
    if len(price) < lookback: return 'none'
    price_subset = price.iloc[-lookback:]
    indicator_subset = indicator.iloc[-lookback:]

    low_indices = argrelextrema(price_subset.values, np.less, order=order)[0]
    high_indices = argrelextrema(price_subset.values, np.greater, order=order)[0]

    if len(low_indices) >= 2:
        if (price_subset.iloc[low_indices[-1]] < price_subset.iloc[low_indices[-2]] and
            indicator_subset.iloc[low_indices[-1]] > indicator_subset.iloc[low_indices[-2]]):
            return 'bullish'

    if len(high_indices) >= 2:
        if (price_subset.iloc[high_indices[-1]] > price_subset.iloc[high_indices[-2]] and
            indicator_subset.iloc[high_indices[-1]] < indicator_subset.iloc[high_indices[-2]]):
            return 'bearish'
    return 'none'

def prepare_dataframe(df, timeframe=PRIMARY_TIMEFRAME):
    if df is None or len(df) < 50:
        print(f"--> Insufficient data for preparation on {timeframe}: {len(df) if df is not None else 0} candles")
        return None
    try:
        # EMA
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=SCALPING_SETTINGS['ema_fast'])
        df['ema_medium'] = ta.trend.ema_indicator(df['close'], window=SCALPING_SETTINGS['ema_medium'])
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=SCALPING_SETTINGS['ema_slow'])
        df['ema_long'] = ta.trend.ema_indicator(df['close'], window=SCALPING_SETTINGS['ema_long'])
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=SCALPING_SETTINGS['rsi_period'])
        # MACD
        macd = ta.trend.MACD(df['close'], window_fast=SCALPING_SETTINGS['macd_fast'], window_slow=SCALPING_SETTINGS['macd_slow'], window_sign=SCALPING_SETTINGS['macd_signal'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=SCALPING_SETTINGS['bb_period'], window_dev=SCALPING_SETTINGS['bb_std'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=SCALPING_SETTINGS['stoch_k'], smooth_window=SCALPING_SETTINGS['stoch_d'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        # ADX
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=SCALPING_SETTINGS['adx_period'])
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        # ATR
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=SCALPING_SETTINGS['atr_period'])
        # Volume
        df['volume_sma'] = df['volume'].rolling(window=SCALPING_SETTINGS['volume_sma_period']).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        # Support & Resistance
        df['resistance'] = df['high'].rolling(window=15).max().shift(1)
        df['support'] = df['low'].rolling(window=15).min().shift(1)
        # Trend detection
        df['trend'] = np.select(
            [ (df['ema_fast'] > df['ema_medium']) & (df['ema_slow'] > df['ema_long']),
              (df['ema_fast'] < df['ema_medium']) & (df['ema_slow'] < df['ema_long']) ],
            ['up', 'down'], default='sideways'
        )
        # Candlestick Patterns
        df['candle_pattern'] = analyze_candlestick_patterns(df)
        # Divergence (calculated on the last candle)
        df['rsi_divergence'] = 'none'
        df['macd_divergence'] = 'none'
        if len(df) > SCALPING_SETTINGS['divergence_lookback']:
            df.loc[df.index[-1], 'rsi_divergence'] = find_divergence(df['close'], df['rsi'], SCALPING_SETTINGS['divergence_lookback'])
            df.loc[df.index[-1], 'macd_divergence'] = find_divergence(df['close'], df['macd'], SCALPING_SETTINGS['divergence_lookback'])

        df.dropna(inplace=True)
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"--> Error preparing dataframe for {timeframe}: {e}")
        traceback.print_exc()
        return None

def quality_filter(symbol, market_stats, df_primary):
    if not market_stats:
        print(f"--> Skipping {symbol}: No market stats available.")
        return False
    # Volume filter
    if market_stats['volume_24h'] < MARKET_FILTERS['min_daily_volume']:
        print(f"--> Skipping {symbol}: Low 24h volume ({market_stats['volume_24h']:,.0f}).")
        return False
    # Price change filter
    abs_change = abs(market_stats['change_rate_24h'])
    if not (MARKET_FILTERS['min_price_change_percent'] <= abs_change <= MARKET_FILTERS['max_price_change_percent']):
        print(f"--> Skipping {symbol}: 24h change ({abs_change:.2f}%) is outside acceptable range.")
        return False
    # Volatility filter (using ATR from prepared dataframe)
    if df_primary is not None and not df_primary.empty:
        volatility_pct = (df_primary['atr'].iloc[-1] / df_primary['close'].iloc[-1]) * 100
        if not (MARKET_FILTERS['min_volatility_percent'] <= volatility_pct <= MARKET_FILTERS['max_volatility_percent']):
            print(f"--> Skipping {symbol}: Volatility ({volatility_pct:.2f}%) is outside acceptable range.")
            return False
    
    print(f"-> {symbol} passed quality filters.")
    return True

def generate_tradingview_link(symbol):
    tv_symbol = symbol.replace('-', '')
    return f"https://www.tradingview.com/chart/?symbol=KUCOIN:{tv_symbol}"

def main():
    print("🚀 Starting High Win-Rate Crypto Analysis...")
    signals_sent = 0
    tehran_tz = pytz.timezone('Asia/Tehran')
    active_signals = {s['symbol']: s for s in load_signals() if s['status'] == 'active'}

    for crypto in CRYPTOCURRENCIES:
        print(f"\n{'='*50}\nAnalyzing {crypto}...")
        try:
            trading_symbol = KUCOIN_SUPPORTED_PAIRS.get(crypto, crypto)
            if trading_symbol != crypto: print(f"-> Using {trading_symbol} for {crypto}")

            market_stats = fetch_market_stats(trading_symbol)
            if not quality_filter(crypto, market_stats, None):
                continue

            if crypto in active_signals:
                created_at = datetime.fromisoformat(active_signals[crypto]['created_at'])
                time_diff_min = (datetime.now(pytz.utc) - created_at).total_seconds() / 60
                if time_diff_min < SCALPING_SETTINGS['signal_cooldown_minutes']:
                    print(f"--> Skipping {crypto}: In cooldown for another {SCALPING_SETTINGS['signal_cooldown_minutes'] - time_diff_min:.0f} minutes.")
                    continue
            
            # Fetch data for all timeframes
            df_primary = fetch_kline_data(trading_symbol, KLINE_SIZE, PRIMARY_TIMEFRAME)
            df_higher = fetch_kline_data(trading_symbol, KLINE_SIZE, HIGHER_TIMEFRAME)
            df_trend = fetch_kline_data(trading_symbol, KLINE_SIZE, TREND_TIMEFRAME)
            if any(df is None for df in [df_primary, df_higher, df_trend]): continue
            
            # Prepare all dataframes
            prepared_primary = prepare_dataframe(df_primary, PRIMARY_TIMEFRAME)
            prepared_higher = prepare_dataframe(df_higher, HIGHER_TIMEFRAME)
            prepared_trend = prepare_dataframe(df_trend, TREND_TIMEFRAME)
            if any(df is None or df.empty for df in [prepared_primary, prepared_higher, prepared_trend]):
                print(f"--> Skipping {crypto}: Failed to prepare one or more dataframes.")
                continue

            if not quality_filter(crypto, market_stats, prepared_primary):
                continue
            
            print(f"-> Generating signals for {crypto}...")
            signals = generate_signals(prepared_primary, prepared_higher, prepared_trend, crypto)
            
            for signal in signals:
                tradingview_link = generate_tradingview_link(signal['symbol'])
                message = (
                    f"<b>🚨 Signal {signal['type']} for {signal['symbol']} 🚨</b>\n\n"
                    f"💰 <b>Entry Price:</b> <code>{signal['entry_price']:.8f}</code>\n"
                    f"🎯 <b>Target Price:</b> <code>{signal['target_price']:.8f}</code>\n"
                    f"🛑 <b>Stop Loss:</b> <code>{signal['stop_loss']:.8f}</code>\n\n"
                    f"🏆 <b>Score:</b> {signal['score']}/100\n"
                    f"📈 <b>Risk/Reward Ratio:</b> {signal['risk_reward_ratio']:.2f}\n\n"
                    f"<b>📝 Reasons:</b>\n{signal['reasons']}\n\n"
                    f"📊 <a href='{tradingview_link}'>TradingView Chart</a>\n"
                    f"⏱️ <b>Time (Tehran):</b> {signal['time']}"
                )
                if send_telegram_message(message):
                    signals_sent += 1
                    save_signal(signal) # The signal with float prices is saved here
                    print(f"✅ Signal sent and saved for {crypto}: {signal['type']}")
                else:
                    print(f"❌ Failed to send signal for {crypto}")

        except Exception as e:
            print(f"--> UNHANDLED ERROR during analysis of {crypto}: {e}")
            traceback.print_exc()
        time.sleep(1)

    print(f"\n{'='*50}\nAnalysis complete. {signals_sent} new signals sent.")
    if signals_sent > 0:
        send_telegram_message(f"✅ Scan finished. Sent {signals_sent} new high-quality signals.", silent=True)

if __name__ == "__main__":
    main()
