# -*- coding: utf-8 -*-

# لیست ارزهای دیجیتال برای بررسی - فقط ارزهای با حجم بالا
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
    # "MATIC-USDT": "POLY-USDT", # Example if needed
}

# تنظیمات استراتژی بهینه شده برای Win Rate بالا
SCALPING_SETTINGS = {
    # RSI - محافظه‌کارانه‌تر
    'rsi_period': 14,
    'rsi_overbought': 75,  # سخت‌تر
    'rsi_oversold': 25,    # سخت‌تر
    'rsi_neutral_zone': (40, 60),  # ناحیه خنثی

    # EMA - برای تشخیص ترند قوی‌تر
    'ema_fast': 8,
    'ema_medium': 21,
    'ema_slow': 50,
    'ema_long': 200,  # برای ترند کلی

    # MACD - حساس‌تر
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,

    # Bollinger Bands
    'bb_period': 20,
    'bb_std': 2.0,
    'bb_squeeze_threshold': 0.015, # برای تشخیص فشردگی و کم نوسانی

    # Stochastic - محافظه‌کارانه‌تر
    'stoch_k': 14,
    'stoch_d': 3,
    'stoch_smooth_k': 3,
    'stoch_overbought': 85,  # سخت‌تر
    'stoch_oversold': 15,    # سخت‌تر

    # ADX - برای قدرت ترند
    'adx_period': 14,
    'adx_strong_trend': 25,    # ترند قوی
    'adx_weak_trend': 20,      # ترند ضعیف یا رنج

    # Volume Analysis
    'volume_sma_period': 20,
    'volume_spike_multiplier': 2.0,

    # Price Action & Risk
    'atr_period': 14,
    'profit_target_multiplier': 2.5,  # هدف بالاتر
    'stop_loss_multiplier': 1.2,     # حد ضرر پویاتر

    # Multi-timeframe
    'higher_tf_confirmation': True,

    # Risk Management - سخت‌تر
    'min_risk_reward_ratio': 2.0,    # R:R بالاتر

    # Signal Quality
    'min_score_threshold': 65,       # آستانه امتیاز بالاتر
    'signal_cooldown_minutes': 120,  # کولداون بیشتر
    'min_factors_required': 2,       # حداقل تعداد فاکتورهای اضافی

    # Divergence
    'divergence_lookback': 25,
}

# تنظیمات تایم فریم‌ها
PRIMARY_TIMEFRAME = "15min"    # تایم فریم اصلی
HIGHER_TIMEFRAME = "1hour"     # تایم فریم بالاتر
TREND_TIMEFRAME = "4hour"      # برای ترند کلی
KLINE_SIZE = 400               # داده بیشتر برای دقت اندیکاتورها
SIGNALS_FILE = "data/signals.json"

# تنظیمات API کوکوین
KUCOIN_BASE_URL = "https://api.kucoin.com"
KUCOIN_KLINE_ENDPOINT = "/api/v1/market/candles"
KUCOIN_TICKER_ENDPOINT = "/api/v1/market/orderbook/level1"
KUCOIN_STATS_ENDPOINT = "/api/v1/market/stats"

# وزن‌دهی جدید برای Win Rate بالا - محافظه‌کارانه‌تر
SIGNAL_WEIGHTS = {
    # فاکتورهای اصلی (الزامی) - امتیاز بالا برای نشان دادن اهمیت
    'trend_alignment': 25,
    'volume_confirmation': 20,
    'multi_tf_confluence': 15,

    # اندیکاتورهای تکنیکال
    'rsi_extreme': 10,
    'macd_momentum': 8,
    'stoch_confirmation': 7,
    'bb_breakout': 12,
    'price_action': 10,

    # فاکتورهای کمکی
    'divergence': 15,
    'support_resistance': 10,
    'candlestick_pattern': 8,
    'adx_strength': 5,
}

# شرایط ورود سخت‌تر
ENTRY_CONDITIONS = {
    'buy': {
        'required_factors': [
            'trend_alignment',
            'volume_confirmation',
            'multi_tf_confluence'
        ],
        'minimum_additional': SCALPING_SETTINGS['min_factors_required'],
        'forbidden_conditions': [
            'counter_trend',
            'low_volume',
            'ranging_market'
        ]
    },
    'sell': {
        'required_factors': [
            'trend_alignment',
            'volume_confirmation',
            'multi_tf_confluence'
        ],
        'minimum_additional': SCALPING_SETTINGS['min_factors_required'],
        'forbidden_conditions': [
            'counter_trend',
            'low_volume',
            'ranging_market'
        ]
    }
}

# فیلترهای کیفیت بازار
MARKET_FILTERS = {
    'min_daily_volume': 1000000,
    'min_price_change_percent': 0.5,
    'max_price_change_percent': 15.0,
    'min_volatility_percent': 0.5, # حداقل نیم درصد نوسان ATR
    'max_volatility_percent': 8.0,  # حداکثر ۸ درصد نوسان ATR
}

# تنظیمات بک‌تست
BACKTEST_SETTINGS = {
    'initial_capital': 10000,
    'position_size_percent': 5,
    'commission_percent': 0.1,
}
