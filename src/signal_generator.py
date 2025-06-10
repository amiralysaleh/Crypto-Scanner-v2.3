# -*- coding: utf-8 -*-

from datetime import datetime
import pytz
from config import SCALPING_SETTINGS, SIGNAL_WEIGHTS

def calculate_score(factors, atr, current_price):
    """Calculate signal score based on weighted factors and volatility."""
    max_score = sum(SIGNAL_WEIGHTS.values()) # Max score can exceed 100 based on weights
    score = 0
    for factor in factors:
        score += SIGNAL_WEIGHTS.get(factor, 0)

    # Adjust score based on volatility (ATR) - Higher volatility slightly reduces score
    volatility_factor = max(0.7, min(1.0, 1.0 - 0.5 * (atr / current_price))) # Less impact
    
    # Scale score to 100
    scaled_score = int((score / max_score) * 100 * volatility_factor)
    return min(scaled_score, 100) # Cap at 100


def generate_signals(df_primary, df_higher, symbol):
    """Generate buy and sell signals with enhanced filters and scoring."""
    # Ensure enough data for Ichimoku Chikou Span check
    ichi_base_period = SCALPING_SETTINGS['ichi_base_period']
    if df_primary is None or len(df_primary) < ichi_base_period + 2:
        print(f"Skipping {symbol}: Not enough primary data for Ichimoku ({len(df_primary)} candles)")
        return []
    if df_higher is None or len(df_higher) < 2:
        print(f"Skipping {symbol}: Not enough higher TF data ({len(df_higher)} candles)")
        return []

    signals = []
    latest = df_primary.iloc[-1]
    prev = df_primary.iloc[-2]
    # Get the price 26 periods ago for Chikou check
    price_26_ago = df_primary['close'].iloc[-1 - ichi_base_period]

    higher_tf_trend = df_higher.iloc[-1]['trend_confirmed']
    current_price = latest['close']
    atr = latest['atr']

    # Check for minimum trend strength
    if latest['adx'] < SCALPING_SETTINGS['adx_threshold']:
        print(f"Skipping {symbol}: ADX ({latest['adx']:.2f}) below threshold ({SCALPING_SETTINGS['adx_threshold']}). Market likely ranging.")
        return []

    # ----- Buy Strategy -----
    buy_factors = set()
    buy_reasons = []

    # RSI
    if latest['rsi'] < SCALPING_SETTINGS['rsi_oversold'] + 5: # Widen zone slightly
        buy_factors.add('rsi')
        buy_reasons.append(f"RSI near oversold ({latest['rsi']:.2f})")

    # Stochastic
    if latest['stoch_k'] < SCALPING_SETTINGS['stoch_oversold'] and latest['stoch_k'] > latest['stoch_d']:
        buy_factors.add('stoch')
        buy_reasons.append(f"Stoch K > D in oversold ({latest['stoch_k']:.2f} > {latest['stoch_d']:.2f})")

    # EMA Cross
    if prev['ema_short'] <= prev['ema_medium'] and latest['ema_short'] > latest['ema_medium']:
        buy_factors.add('ema')
        buy_reasons.append("EMA Short/Medium bullish cross")

    # MACD Cross
    if prev['macd'] <= prev['macd_signal'] and latest['macd'] > latest['macd_signal']:
        buy_factors.add('macd')
        buy_reasons.append("MACD bullish cross")

    # Bollinger Bands
    if latest['close'] <= latest['bb_lower']:
        buy_factors.add('bb')
        buy_reasons.append("Price at/below lower BB")

    # Support
    if abs(latest['close'] - latest['support']) / latest['close'] < 0.01: # Within 1% of support
        buy_factors.add('support')
        buy_reasons.append(f"Price near support ({latest['support']:.4f})")
        
    # Volume
    if latest['volume_change'] > SCALPING_SETTINGS['volume_change_threshold']:
        buy_factors.add('volume')
        buy_reasons.append(f"Volume surge ({latest['volume_change']:.2f}X)")

    # Ichimoku Cloud (Bullish Signals) - CORRECTED CHIKOU CHECK
    ichi_bullish = False
    if (latest['close'] > latest['ichi_a'] and latest['close'] > latest['ichi_b'] and # Price above cloud
        latest['ichi_conv'] > latest['ichi_base'] and # Tenkan > Kijun
        latest['close'] > price_26_ago): # Chikou (Current Price) > Price 26 periods ago
        buy_factors.add('ichi')
        buy_reasons.append("Strong Ichimoku bullish (Price > Cloud, TK Cross, Chikou > Price_ago)")
        ichi_bullish = True
    elif (latest['close'] > latest['ichi_a'] and latest['close'] > latest['ichi_b'] and
          prev['ichi_conv'] <= prev['ichi_base'] and latest['ichi_conv'] > latest['ichi_base']):
        buy_factors.add('ichi')
        buy_reasons.append("Ichimoku TK bullish cross above cloud")
        ichi_bullish = True

    # Candlestick Pattern
    if latest['candle_pattern'] == 'bullish_engulfing' or latest['candle_pattern'] == 'hammer':
        buy_factors.add('candle')
        buy_reasons.append(f"Bullish candle pattern ({latest['candle_pattern']})")

    # Divergence
    if latest['rsi_divergence'] == 'bullish' or latest['macd_divergence'] == 'bullish':
        buy_factors.add('divergence')
        buy_reasons.append(f"Bullish Divergence detected (RSI: {latest['rsi_divergence']}, MACD: {latest['macd_divergence']})")

    # Higher TF Trend
    if higher_tf_trend == 'up':
        buy_factors.add('higher_tf')
        buy_reasons.append("Higher TF (4h) is in uptrend")
        
    # ADX (Confirming Bullish Momentum)
    if latest['adx_pos'] > latest['adx_neg']:
        buy_factors.add('adx')
        buy_reasons.append(f"ADX shows bullish momentum (+DI > -DI)")

    # Generate Buy Signal
    if len(buy_reasons) >= 3: # Require at least 3 factors
        score = calculate_score(buy_factors, atr, current_price)
        if score >= SCALPING_SETTINGS['min_score_threshold']:
            target_price = current_price + (atr * SCALPING_SETTINGS['profit_target_multiplier'])
            stop_loss = current_price - (atr * SCALPING_SETTINGS['stop_loss_multiplier'])
            if (current_price - stop_loss) == 0: # Avoid division by zero
                risk_reward_ratio = 999
            else:
                risk_reward_ratio = (target_price - current_price) / (current_price - stop_loss)

            if risk_reward_ratio >= SCALPING_SETTINGS['min_risk_reward_ratio']:
                current_time_obj = datetime.now(pytz.timezone('Asia/Tehran'))
                current_time = current_time_obj.strftime("%Y-%m-%d %H:%M:%S")
                signals.append({
                    'symbol': symbol, 'type': 'BUY',
                    'current_price': f"{current_price:.8f}", 'target_price': f"{target_price:.8f}",
                    'stop_loss': f"{stop_loss:.8f}", 'time': current_time,
                    'reasons': "\n".join([f"✅ {reason}" for reason in buy_reasons]),
                    'score': score, 'status': 'active',
                    'created_at': current_time_obj.isoformat(),
                    'risk_reward_ratio': risk_reward_ratio
                })

    # ----- Sell Strategy -----
    sell_factors = set()
    sell_reasons = []

    # RSI
    if latest['rsi'] > SCALPING_SETTINGS['rsi_overbought'] - 5: # Widen zone slightly
        sell_factors.add('rsi')
        sell_reasons.append(f"RSI near overbought ({latest['rsi']:.2f})")

    # Stochastic
    if latest['stoch_k'] > SCALPING_SETTINGS['stoch_overbought'] and latest['stoch_k'] < latest['stoch_d']:
        sell_factors.add('stoch')
        sell_reasons.append(f"Stoch K < D in overbought ({latest['stoch_k']:.2f} < {latest['stoch_d']:.2f})")

    # EMA Cross
    if prev['ema_short'] >= prev['ema_medium'] and latest['ema_short'] < latest['ema_medium']:
        sell_factors.add('ema')
        sell_reasons.append("EMA Short/Medium bearish cross")

    # MACD Cross
    if prev['macd'] >= prev['macd_signal'] and latest['macd'] < latest['macd_signal']:
        sell_factors.add('macd')
        sell_reasons.append("MACD bearish cross")

    # Bollinger Bands
    if latest['close'] >= latest['bb_upper']:
        sell_factors.add('bb')
        sell_reasons.append("Price at/above upper BB")

    # Resistance
    if abs(latest['close'] - latest['resistance']) / latest['close'] < 0.01: # Within 1% of resistance
        sell_factors.add('resistance')
        sell_reasons.append(f"Price near resistance ({latest['resistance']:.4f})")
        
    # Ichimoku Cloud (Bearish Signals) - CORRECTED CHIKOU CHECK
    ichi_bearish = False
    if (latest['close'] < latest['ichi_a'] and latest['close'] < latest['ichi_b'] and # Price below cloud
        latest['ichi_conv'] < latest['ichi_base'] and # Tenkan < Kijun
        latest['close'] < price_26_ago): # Chikou (Current Price) < Price 26 periods ago
        sell_factors.add('ichi')
        sell_reasons.append("Strong Ichimoku bearish (Price < Cloud, TK Cross, Chikou < Price_ago)")
        ichi_bearish = True
    elif (latest['close'] < latest['ichi_a'] and latest['close'] < latest['ichi_b'] and
          prev['ichi_conv'] >= prev['ichi_base'] and latest['ichi_conv'] < latest['ichi_base']):
        sell_factors.add('ichi')
        sell_reasons.append("Ichimoku TK bearish cross below cloud")
        ichi_bearish = True

    # Candlestick Pattern
    if latest['candle_pattern'] == 'bearish_engulfing' or latest['candle_pattern'] == 'shooting_star':
        sell_factors.add('candle')
        sell_reasons.append(f"Bearish candle pattern ({latest['candle_pattern']})")

    # Divergence
    if latest['rsi_divergence'] == 'bearish' or latest['macd_divergence'] == 'bearish':
        sell_factors.add('divergence')
        sell_reasons.append(f"Bearish Divergence detected (RSI: {latest['rsi_divergence']}, MACD: {latest['macd_divergence']})")

    # Higher TF Trend
    if higher_tf_trend == 'down':
        sell_factors.add('higher_tf')
        sell_reasons.append("Higher TF (4h) is in downtrend")
        
    # ADX (Confirming Bearish Momentum)
    if latest['adx_neg'] > latest['adx_pos']:
        sell_factors.add('adx')
        sell_reasons.append(f"ADX shows bearish momentum (-DI > +DI)")

    # Generate Sell Signal
    if len(sell_reasons) >= 3: # Require at least 3 factors
        score = calculate_score(sell_factors, atr, current_price)
        if score >= SCALPING_SETTINGS['min_score_threshold']:
            target_price = current_price - (atr * SCALPING_SETTINGS['profit_target_multiplier'])
            stop_loss = current_price + (atr * SCALPING_SETTINGS['stop_loss_multiplier'])
            if (stop_loss - current_price) == 0: # Avoid division by zero
                risk_reward_ratio = 999
            else:
                risk_reward_ratio = (current_price - target_price) / (stop_loss - current_price)

            if risk_reward_ratio >= SCALPING_SETTINGS['min_risk_reward_ratio']:
                current_time_obj = datetime.now(pytz.timezone('Asia/Tehran'))
                current_time = current_time_obj.strftime("%Y-%m-%d %H:%M:%S")
                signals.append({
                    'symbol': symbol, 'type': 'SELL',
                    'current_price': f"{current_price:.8f}", 'target_price': f"{target_price:.8f}",
                    'stop_loss': f"{stop_loss:.8f}", 'time': current_time,
                    'reasons': "\n".join([f"✅ {reason}" for reason in sell_reasons]),
                    'score': score, 'status': 'active',
                    'created_at': current_time_obj.isoformat(),
                    'risk_reward_ratio': risk_reward_ratio
                })

    return signals
