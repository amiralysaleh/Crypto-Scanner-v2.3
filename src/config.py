# -*- coding: utf-8 -*-

# لیست ارزهای دیجیتال برای بررسی - نمادهای سازگار با KuCoin

CRYPTOCURRENCIES = [
“BTC-USDT”, “ETH-USDT”, “BNB-USDT”, “SOL-USDT”, “XRP-USDT”,
“ADA-USDT”, “DOGE-USDT”, “SHIB-USDT”, “DOT-USDT”,
“LTC-USDT”, “AVAX-USDT”, “LINK-USDT”, “UNI-USDT”, “ATOM-USDT”,
“TRX-USDT”, “NEAR-USDT”, “MATIC-USDT”, “APT-USDT”,
“PEPE-USDT”, “ICP-USDT”, “ETC-USDT”, “XLM-USDT”, “HBAR-USDT”,
“INJ-USDT”, “VET-USDT”, “CRO-USDT”, “OP-USDT”, “ALGO-USDT”,
“GRT-USDT”, “SUI-USDT”, “AAVE-USDT”, “FTM-USDT”, “FLOW-USDT”,
“AR-USDT”, “EGLD-USDT”, “AXS-USDT”, “CHZ-USDT”, “SAND-USDT”,
“MANA-USDT”, “NEO-USDT”, “KAVA-USDT”, “XTZ-USDT”,
“MINA-USDT”, “GALA-USDT”, “ZIL-USDT”, “ENJ-USDT”, “1INCH-USDT”,
“TON-USDT”, “COMP-USDT”, “ZEC-USDT”, “DASH-USDT”, “LRC-USDT”,
“QTUM-USDT”, “ICX-USDT”, “ONT-USDT”, “WAVES-USDT”, “KSM-USDT”,
“CHR-USDT”, “ANKR-USDT”, “OCEAN-USDT”, “IOST-USDT”, “HONEY-USDT”,
“RSR-USDT”, “DCR-USDT”, “SYS-USDT”, “GLMR-USDT”, “BICO-USDT”,
“COTI-USDT”, “SKL-USDT”, “BAL-USDT”, “LPT-USDT”, “CELR-USDT”,
“DGB-USDT”, “XAI-USDT”, “API3-USDT”, “OMG-USDT”, “POWR-USDT”,
“SXP-USDT”, “REQ-USDT”, “NKN-USDT”, “CTSI-USDT”,
“HYPER-USDT”, “FLUX-USDT”, “AUDIO-USDT”,
“CVC-USDT”, “SNT-USDT”, “BCH-USDT”, “XMR-USDT”, “EOS-USDT”,
“TAO-USDT”, “PYTH-USDT”, “AERGO-USDT”, “KLAY-USDT”, “TRAC-USDT”,
“LTO-USDT”, “MLN-USDT”, “RIF-USDT”, “GHST-USDT”, “DUSK-USDT”,
“BAND-USDT”, “ORBS-USDT”, “UOS-USDT”, “ERN-USDT”, “MDT-USDT”,
“KMD-USDT”, “WNCG-USDT”, “QKC-USDT”, “FIL-USDT”, “ZRX-USDT”,
“SNX-USDT”, “REN-USDT”, “BNT-USDT”, “STMX-USDT”, “MTL-USDT”,
“SUSHI-USDT”, “LUNA-USDT”, “RUNE-USDT”, “DYDX-USDT”, “YFI-USDT”,
“CRV-USDT”, “UMA-USDT”, “FET-USDT”, “RAY-USDT”, “AKRO-USDT”,
“CKB-USDT”, “ALPHA-USDT”, “PERP-USDT”, “LIT-USDT”, “CTK-USDT”,
“BADGER-USDT”, “C98-USDT”, “DODO-USDT”, “ELF-USDT”, “FRONT-USDT”,
“GTC-USDT”, “HNT-USDT”, “IDEX-USDT”, “JASMY-USDT”, “KDA-USDT”,
“LINA-USDT”, “MIR-USDT”, “OGN-USDT”, “POLS-USDT”, “QNT-USDT”,
“REEF-USDT”, “SFP-USDT”, “TOMO-USDT”, “SPELL-USDT”, “ILV-USDT”,
“MOVR-USDT”, “GLM-USDT”, “ARB-USDT”, “IMX-USDT”
]

# جایگزین کردن نمادهای غیر پشتیبانی شده

KUCOIN_SUPPORTED_PAIRS = {
“MATIC-USDT”: “POLY-USDT”,
}

# تنظیمات استراتژی سوینگ تریدینگ با سیستم امتیازدهی بهبود یافته - هدف وین ریت 80%+

SCALPING_SETTINGS = {
# RSI - تنظیمات محافظه کارانه برای سیگنالهای قوی تر
‘rsi_period’: 21,
‘rsi_overbought’: 75,
‘rsi_oversold’: 25,
# EMA - سیستم سه گانه برای تایید ترند
‘ema_short’: 21,
‘ema_medium’: 50,
‘ema_long’: 200,
# MACD - تنظیمات استاندارد برای 1 ساعته
‘macd_fast’: 12,
‘macd_slow’: 26,
‘macd_signal’: 9,
# Bollinger Bands - تنظیمات محافظه کارانه
‘bb_period’: 20,
‘bb_std’: 2.5,
# Stochastic - فیلتر قوی تر
‘stoch_k’: 21,
‘stoch_d’: 5,
‘stoch_smooth_k’: 3,
‘stoch_overbought’: 85,
‘stoch_oversold’: 15,
# ADX - تایید قوی ترند
‘adx_period’: 14,
‘adx_threshold’: 30,
# Ichimoku - تنظیمات کلاسیک
‘ichi_conv_period’: 9,
‘ichi_base_period’: 26,
‘ichi_span_b_period’: 52,
‘ichi_lag_span_period’: 26,
# Divergence
‘divergence_lookback’: 50,
# General - تنظیمات محافظه کارانه برای کیفیت بالا
‘min_volume_threshold’: 1000000,
‘volume_change_threshold’: 1.8,
‘profit_target_multiplier’: 3.5,
‘stop_loss_multiplier’: 1.2,
‘min_score_threshold’: 40,
‘min_risk_reward_ratio’: 2.5,
‘signal_cooldown_minutes’: 480,
‘max_signals_per_symbol’: 1,
‘trend_confirmation_window’: 20,
‘fee_percent’: 0.1,
}

# تنظیمات تایم فریم‌ها - تغییر به 1 ساعته

PRIMARY_TIMEFRAME = “1hour”
HIGHER_TIMEFRAME = “4hour”
KLINE_SIZE = 500
SIGNALS_FILE = “data/signals.json”

# تنظیمات API کوکوین

KUCOIN_BASE_URL = “https://api.kucoin.com”
KUCOIN_KLINE_ENDPOINT = “/api/v1/market/candles”
KUCOIN_TICKER_ENDPOINT = “/api/v1/market/orderbook/level1”
KUCOIN_STATS_ENDPOINT = “/api/v1/market/stats”

# وزن‌دهی فاکتورها برای سیستم امتیازدهی - تاکید بر کیفیت

SIGNAL_WEIGHTS = {
‘rsi’: 15, ‘ema’: 20, ‘macd’: 20, ‘bb’: 10,
‘stoch’: 12, ‘adx’: 15, ‘ichi’: 25, ‘divergence’: 30,
‘candle’: 18, ‘volume’: 8, ‘support’: 12, ‘resistance’: 12,
‘higher_tf’: 25
}