# -*- coding: utf-8 -*-

from datetime import datetime
import pytz
from config import SCALPING_SETTINGS, SIGNAL_WEIGHTS, ENTRY_CONDITIONS

def calculate_score(factors):
    """Calculate signal score based on weighted factors."""
    score = sum(SIGNAL_WEIGHTS.get(factor, 0) for factor in factors)
    max_possible_score = sum(SIGNAL_WEIGHTS.values())
    # Scale score to be out of 100
    scaled_score = int((score / max_possible_score) * 100)
    return min(scaled_score, 100)

def get_signal_reasons(factors, reasons_map):
    """Format reasons for the telegram message."""
    return "\n".join([f"✅ {reasons_map[factor]}" for factor in factors if factor in reasons_map])

def generate_signals(df_primary, df_higher, df_trend, symbol):
    """
    Generate buy and sell signals based on the high win-rate strategy.
    This function implements the strict entry conditions from the config.
    """
    signals = []
    
    latest_p = df_primary.iloc[-1]
    latest_h = df_higher.iloc[-1]
    latest_t = df_trend.iloc[-1]

    current_price = latest_p['close']
    atr = latest_p['atr']
    if atr == 0: return []

    is_trend_aligned_up = latest_p['trend'] == 'up' and latest_h['trend'] == 'up' and latest_t['trend'] == 'up'
    is_trend_aligned_down = latest_p['trend'] == 'down' and latest_h['trend'] == 'down' and latest_t['trend'] == 'down'

    # --- BUY SIGNAL LOGIC ---
    if is_trend_aligned_up:
        buy_factors = set()
        buy_reasons_map = {}

        buy_factors.add('trend_alignment')
        buy_reasons_map['trend_alignment'] = f"Trend Alignment (P: {latest_p['trend']}, H: {latest_h['trend']}, T: {latest_t['trend']})"

        if latest_p['volume_ratio'] >= SCALPING_SETTINGS['volume_spike_multiplier']:
            buy_factors.add('volume_confirmation')
            buy_reasons_map['volume_confirmation'] = f"Volume Spike ({latest_p['volume_ratio']:.2f}x SMA)"

        if latest_h['rsi'] > 50 and latest_t['rsi'] > 50:
            buy_factors.add('multi_tf_confluence')
            buy_reasons_map['multi_tf_confluence'] = "Multi-TF RSI Confirmation"
        
        if all(factor in buy_factors for factor in ENTRY_CONDITIONS['buy']['required_factors']):
            additional_factors = set()
            if latest_p['rsi'] < SCALPING_SETTINGS['rsi_oversold']:
                additional_factors.add('rsi_extreme')
                buy_reasons_map['rsi_extreme'] = f"Primary RSI Oversold ({latest_p['rsi']:.2f})"
            if latest_p['macd'] > latest_p['macd_signal']:
                additional_factors.add('macd_momentum')
                buy_reasons_map['macd_momentum'] = "MACD Bullish Momentum"
            if latest_p['stoch_k'] < SCALPING_SETTINGS['stoch_oversold'] and latest_p['stoch_k'] > latest_p['stoch_d']:
                additional_factors.add('stoch_confirmation')
                buy_reasons_map['stoch_confirmation'] = "Stochastic Bullish Cross"
            if df_primary.iloc[-2]['bb_width'] < SCALPING_SETTINGS['bb_squeeze_threshold'] and latest_p['close'] > latest_p['bb_upper']:
                 additional_factors.add('bb_breakout')
                 buy_reasons_map['bb_breakout'] = "Bollinger Bands Breakout from Squeeze"
            if latest_p['rsi_divergence'] == 'bullish' or latest_p['macd_divergence'] == 'bullish':
                additional_factors.add('divergence')
                buy_reasons_map['divergence'] = "Bullish Divergence Detected"
            if abs(current_price - latest_p['support']) / current_price < 0.01:
                additional_factors.add('support_resistance')
                buy_reasons_map['support_resistance'] = "Price near key support"
            if latest_p['candle_pattern'] in ['bullish_engulfing', 'hammer']:
                 additional_factors.add('candlestick_pattern')
                 buy_reasons_map['candlestick_pattern'] = f"Bullish Pattern ({latest_p['candle_pattern']})"

            if len(additional_factors) >= ENTRY_CONDITIONS['buy']['minimum_additional']:
                all_factors = buy_factors.union(additional_factors)
                score = calculate_score(all_factors)
                if score >= SCALPING_SETTINGS['min_score_threshold']:
                    target_price = current_price + (atr * SCALPING_SETTINGS['profit_target_multiplier'])
                    stop_loss = current_price - (atr * SCALPING_SETTINGS['stop_loss_multiplier'])
                    if (current_price - stop_loss) == 0: return []
                    risk_reward_ratio = (target_price - current_price) / (current_price - stop_loss)

                    if risk_reward_ratio >= SCALPING_SETTINGS['min_risk_reward_ratio']:
                        current_time_obj = datetime.now(pytz.utc)
                        signals.append({
                            'symbol': symbol, 'type': 'BUY',
                            'entry_price': current_price,
                            'target_price': target_price,
                            'stop_loss': stop_loss,
                            'time': current_time_obj.astimezone(pytz.timezone('Asia/Tehran')).strftime("%Y-%m-%d %H:%M:%S"),
                            'reasons': get_signal_reasons(all_factors, buy_reasons_map),
                            'score': score, 'status': 'active',
                            'created_at': current_time_obj.isoformat(),
                            'risk_reward_ratio': risk_reward_ratio
                        })

    # --- SELL SIGNAL LOGIC ---
    if is_trend_aligned_down:
        sell_factors = set()
        sell_reasons_map = {}

        sell_factors.add('trend_alignment')
        sell_reasons_map['trend_alignment'] = f"Trend Alignment (P: {latest_p['trend']}, H: {latest_h['trend']}, T: {latest_t['trend']})"

        if latest_p['volume_ratio'] >= SCALPING_SETTINGS['volume_spike_multiplier']:
            sell_factors.add('volume_confirmation')
            sell_reasons_map['volume_confirmation'] = f"Volume Spike ({latest_p['volume_ratio']:.2f}x SMA)"

        if latest_h['rsi'] < 50 and latest_t['rsi'] < 50:
            sell_factors.add('multi_tf_confluence')
            sell_reasons_map['multi_tf_confluence'] = "Multi-TF RSI Confirmation"

        if all(factor in sell_factors for factor in ENTRY_CONDITIONS['sell']['required_factors']):
            additional_factors = set()
            if latest_p['rsi'] > SCALPING_SETTINGS['rsi_overbought']:
                additional_factors.add('rsi_extreme')
                sell_reasons_map['rsi_extreme'] = f"Primary RSI Overbought ({latest_p['rsi']:.2f})"
            if latest_p['macd'] < latest_p['macd_signal']:
                additional_factors.add('macd_momentum')
                sell_reasons_map['macd_momentum'] = "MACD Bearish Momentum"
            if latest_p['stoch_k'] > SCALPING_SETTINGS['stoch_overbought'] and latest_p['stoch_k'] < latest_p['stoch_d']:
                additional_factors.add('stoch_confirmation')
                sell_reasons_map['stoch_confirmation'] = "Stochastic Bearish Cross"
            if df_primary.iloc[-2]['bb_width'] < SCALPING_SETTINGS['bb_squeeze_threshold'] and latest_p['close'] < latest_p['bb_lower']:
                 additional_factors.add('bb_breakout')
                 sell_reasons_map['bb_breakout'] = "Bollinger Bands Breakout from Squeeze"
            if latest_p['rsi_divergence'] == 'bearish' or latest_p['macd_divergence'] == 'bearish':
                additional_factors.add('divergence')
                sell_reasons_map['divergence'] = "Bearish Divergence Detected"
            if abs(current_price - latest_p['resistance']) / current_price < 0.01:
                additional_factors.add('support_resistance')
                sell_reasons_map['support_resistance'] = "Price near key resistance"
            if latest_p['candle_pattern'] in ['bearish_engulfing', 'shooting_star']:
                 additional_factors.add('candlestick_pattern')
                 sell_reasons_map['candlestick_pattern'] = f"Bearish Pattern ({latest_p['candle_pattern']})"

            if len(additional_factors) >= ENTRY_CONDITIONS['sell']['minimum_additional']:
                all_factors = sell_factors.union(additional_factors)
                score = calculate_score(all_factors)
                if score >= SCALPING_SETTINGS['min_score_threshold']:
                    target_price = current_price - (atr * SCALPING_SETTINGS['profit_target_multiplier'])
                    stop_loss = current_price + (atr * SCALPING_SETTINGS['stop_loss_multiplier'])
                    if (stop_loss - current_price) == 0: return []
                    risk_reward_ratio = (current_price - target_price) / (stop_loss - current_price)

                    if risk_reward_ratio >= SCALPING_SETTINGS['min_risk_reward_ratio']:
                        current_time_obj = datetime.now(pytz.utc)
                        signals.append({
                            'symbol': symbol, 'type': 'SELL',
                            'entry_price': current_price,
                            'target_price': target_price,
                            'stop_loss': stop_loss,
                            'time': current_time_obj.astimezone(pytz.timezone('Asia/Tehran')).strftime("%Y-%m-%d %H:%M:%S"),
                            'reasons': get_signal_reasons(all_factors, sell_reasons_map),
                            'score': score, 'status': 'active',
                            'created_at': current_time_obj.isoformat(),
                            'risk_reward_ratio': risk_reward_ratio
                        })

    return signals
