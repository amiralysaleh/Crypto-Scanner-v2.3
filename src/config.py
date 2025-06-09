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
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    # EMA
    'ema_short': 20,  # تنظیم برای کراس کوتاه‌مدت
    'ema_medium': 50, # تنظیم برای تأیید روند
    'ema_long': 200,  # حفظ برای تحلیل بلندمدت
    # MACD
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    # Bollinger Bands
    'bb_period': 20,
    'bb_std': 2.0,
    # Stochastic
    'stoch_k': 14,
    'stoch_d': 3,
    'stoch_smooth_k': 3,
    'stoch_overbought': 80,
    'stoch_oversold': 20,
    # ADX
    'adx_period': 14,
    'adx_threshold': 25,  # افزایش برای تأیید قوی‌تر روند
    # Ichimoku
    'ichi_conv_period': 9,
    'ichi_base_period': 26,
    'ichi_span_b_period': 52,
    'ichi_lag_span_period': 26,
    # Divergence
    'divergence_lookback': 25,
    # General
    'min_volume_threshold': 500000,
    'volume_change_threshold': 1.5,
    'profit_target_multiplier': 3.0,  # افزایش برای ریسک به ریوارد بهتر
    'stop_loss_multiplier': 1.0,
    'min_score_threshold': 40,  # افزایش برای سیگنال‌های قوی‌تر
    'min_risk_reward_ratio': 3.0,  # افزایش برای وین ریت بالا
    'signal_cooldown_minutes': 1440,  # تغییر به 24 ساعت (1440 دقیقه)
    'max_signals_per_symbol': 1,
    'trend_confirmation_window': 20,
    'fee_percent': 0.1,
}

# تنظیمات تایم فریم‌ها
PRIMARY_TIMEFRAME = "1hour"
HIGHER_TIMEFRAME = "4hour"
KLINE_SIZE = 900
SIGNALS_FILE = "data/signals.json"

# تنظیمات API کوکوین
KUCOIN_BASE_URL = "https://api.kucoin.com"
KUCOIN_KLINE_ENDPOINT = "/api/v1/market/candles"
KUCOIN_TICKER_ENDPOINT = "/api/v1/market/orderbook/level1"
KUCOIN_STATS_ENDPOINT = "/api/v1/market/stats"

# وزن‌دهی فاکتورها برای سیستم امتیازدهی جدید
SIGNAL_WEIGHTS = {
    'rsi': 10,
    'ema': 40,  # افزایش وزن برای کراس EMA (استراتژی اصلی)
    'macd': 15,
    'bb': 0,    # کاهش به 0 برای تمرکز روی EMA
    'stoch': 5,
    'adx': 20,  # افزایش وزن برای تأیید روند
    'ichi': 0,  # کاهش به 0
    'divergence': 0,  # کاهش به 0
    'candle': 0,  # کاهش به 0
    'volume': 5,
    'support': 20,  # افزایش وزن برای نقاط ورود
    'resistance': 20,  # افزایش وزن برای نقاط خروج
    'higher_tf': 15,  # افزایش وزن برای تأیید تایم فریم بالاتر
}