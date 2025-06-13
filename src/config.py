# -*- coding: utf-8 -*-

# لیست ارزهای دیجیتال برای بررسی - نمادهای سازگار با KuCoin
CRYPTOCURRENCIES = [
    "BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT", "XRP-USDT",
    "ADA-USDT", "DOGE-USDT", "SHIB-USDT", "DOT-USDT",
    "LTC-USDT", "AVAX-USDT", "LINK-USDT", "UNI-USDT", "ATOM-USDT",
    "TRX-USDT", "NEAR-USDT", "MATIC-USDT", "APT-USDT",
    "PEPE-USDT", "ICP-USDT", "ETC-USDT", "XLM-USDT", "HBAR-USDT",
    "INJ-USDT", "VET-USDT", "CRO-USDT", "OP-USDT", "ALGO-USDT",
    "GRT-USDT", "SUI-USDT", "AAVE-USDT", "FTM-USDT", "FLOW-USDT",
    "AR-USDT", "EGLD-USDT", "AXS-USDT", "CHZ-USDT", "SAND-USDT",
    "MANA-USDT", "NEO-USDT", "KAVA-USDT", "XTZ-USDT",
    "MINA-USDT", "GALA-USDT", "ZIL-USDT", "ENJ-USDT", "1INCH-USDT",
    "TON-USDT", "COMP-USDT", "ZEC-USDT", "DASH-USDT", "LRC-USDT",
    "QTUM-USDT", "ICX-USDT", "ONT-USDT", "WAVES-USDT", "KSM-USDT",
    "CHR-USDT", "ANKR-USDT", "OCEAN-USDT", "IOST-USDT", "HONEY-USDT",
    "RSR-USDT", "DCR-USDT", "SYS-USDT", "GLMR-USDT", "BICO-USDT",
    "COTI-USDT", "SKL-USDT", "BAL-USDT", "LPT-USDT", "CELR-USDT",
    "DGB-USDT", "XAI-USDT", "API3-USDT", "OMG-USDT", "POWR-USDT",
    "SXP-USDT", "REQ-USDT", "NKN-USDT", "CTSI-USDT",
    "HYPER-USDT", "FLUX-USDT", "AUDIO-USDT",
    "CVC-USDT", "SNT-USDT", "BCH-USDT", "XMR-USDT", "EOS-USDT",
    "TAO-USDT", "PYTH-USDT", "AERGO-USDT", "KLAY-USDT", "TRAC-USDT",
    "LTO-USDT", "MLN-USDT", "RIF-USDT", "GHST-USDT", "DUSK-USDT",
    "BAND-USDT", "ORBS-USDT", "UOS-USDT", "ERN-USDT", "MDT-USDT",
    "KMD-USDT", "WNCG-USDT", "QKC-USDT", "FIL-USDT", "ZRX-USDT",
    "SNX-USDT", "REN-USDT", "BNT-USDT", "STMX-USDT", "MTL-USDT",
    "SUSHI-USDT", "LUNA-USDT", "RUNE-USDT", "DYDX-USDT", "YFI-USDT",
    "CRV-USDT", "UMA-USDT", "FET-USDT", "RAY-USDT", "AKRO-USDT",
    "CKB-USDT", "ALPHA-USDT", "PERP-USDT", "LIT-USDT", "CTK-USDT",
    "BADGER-USDT", "C98-USDT", "DODO-USDT", "ELF-USDT", "FRONT-USDT",
    "GTC-USDT", "HNT-USDT", "IDEX-USDT", "JASMY-USDT", "KDA-USDT",
    "LINA-USDT", "MIR-USDT", "OGN-USDT", "POLS-USDT", "QNT-USDT",
    "REEF-USDT", "SFP-USDT", "TOMO-USDT", "SPELL-USDT", "ILV-USDT",
    "MOVR-USDT", "GLM-USDT", "ARB-USDT", "IMX-USDT"
]
# جایگزین کردن نمادهای غیر پشتیبانی شده
KUCOIN_SUPPORTED_PAIRS = {
    "MATIC-USDT": "POLY-USDT",
}

# تنظیمات استراتژی اسکالپینگ با سیستم امتیازدهی بهبود یافته
SCALPING_SETTINGS = {
    # RSI
    'rsi_period': 10,
    'rsi_overbought': 75,
    'rsi_oversold': 25,
    # EMA
    'ema_short': 10,  # Slightly adjusted
    'ema_medium': 50,
    'ema_long': 100,  # Adjusted
    # MACD
    'macd_fast': 8,
    'macd_slow': 21,
    'macd_signal': 5,
    # Bollinger Bands
    'bb_period': 20,
    'bb_std': 2.0,
    # Stochastic
    'stoch_k': 9,
    'stoch_d': 3,
    'stoch_smooth_k': 1,
    'stoch_overbought': 85,
    'stoch_oversold': 15,
    # ADX
    'adx_period': 14,
    'adx_threshold': 25,  # Minimum trend strength
    # Ichimoku
    'ichi_conv_period': 9,
    'ichi_base_period': 26,
    'ichi_span_b_period': 52,
    'ichi_lag_span_period': 26,
    # Divergence
    'divergence_lookback': 25, # How many candles to look back for divergence
    # General
    'min_volume_threshold': 300000,
    'volume_change_threshold': 1.3,
    'profit_target_multiplier': 2.0,  # Increased R:R target
    'stop_loss_multiplier': 1.0,     # Tighter stop loss
    'min_score_threshold': 60,       # Increased threshold for higher quality
    'min_risk_reward_ratio': 1.7,    # Increased R:R
    'signal_cooldown_minutes': 60,
    'max_signals_per_symbol': 1,
    'trend_confirmation_window': 10,
    'fee_percent': 0.1,
}

# تنظیمات تایم فریم‌ها
PRIMARY_TIMEFRAME = "1hour"
HIGHER_TIMEFRAME = "4hour"
KLINE_SIZE = 800  # Ensure enough data for Ichimoku and lookbacks
SIGNALS_FILE = "data/signals.json"

# تنظیمات API کوکوین
KUCOIN_BASE_URL = "https://api.kucoin.com"
KUCOIN_KLINE_ENDPOINT = "/api/v1/market/candles"
KUCOIN_TICKER_ENDPOINT = "/api/v1/market/orderbook/level1"
KUCOIN_STATS_ENDPOINT = "/api/v1/market/stats"

# وزن‌دهی فاکتورها برای سیستم امتیازدهی جدید
SIGNAL_WEIGHTS = {
    'rsi': 5, 'ema': 5, 'macd': 5, 'bb': 40,
    'stoch': 5, 'adx': 5, 'ichi': 20, 'divergence': 5,
    'candle': 5, 'volume': 5, 'support': 5, 'resistance': 5,
    'higher_tf': 5
}
