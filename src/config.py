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

# تنظیمات استراتژی سوینگ تریدینگ کوتاه‌مدت (1 ساعته) با وین ریت هدف بالای 80%
SCALPING_SETTINGS = {
    # RSI
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    # EMA (برای استراتژی Moving Average Crossover کوتاه‌مدت)
    'ema_short': 20,  # EMA کوتاه برای تغییرات سریع
    'ema_long': 50,   # EMA بلندتر برای تأیید روند
    # MACD (برای تأیید مومنتوم)
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    # Bollinger Bands (برای شناسایی نقاط برگشت)
    'bb_period': 20,
    'bb_std': 2.0,
    # ADX (برای تأیید قدرت روند)
    'adx_period': 14,
    'adx_threshold': 25,  # حداقل قدرت روند
    # حجم
    'min_volume_threshold': 500000,  # حداقل حجم 24 ساعته
    'volume_change_threshold': 1.5,  # افزایش حداقل 50% در حجم
    # مدیریت ریسک
    'profit_target_multiplier': 3.0,  # نسبت سود به ریسک 3:1
    'stop_loss_multiplier': 1.0,     # توقف ضرر بر اساس ATR
    'min_score_threshold': 70,       # آستانه امتیاز بالاتر برای وین ریت بالا
    'min_risk_reward_ratio': 3.0,    # حداقل نسبت ریسک به ریوارد
    'signal_cooldown_hours': 24,     # خنک‌سازی سیگنال برای 24 ساعت
    'max_signals_per_symbol': 1,     # حداکثر یک سیگنال فعال
    'trend_confirmation_window': 20, # پنجره تأیید روند
    'fee_percent': 0.1,
}

# تنظیمات تایم فریم‌ها
PRIMARY_TIMEFRAME = "1hour"  # تایم فریم اصلی
HIGHER_TIMEFRAME = "4hour"  # تایم فریم بالاتر برای تأیید روند
KLINE_SIZE = 500  # داده‌های کافی برای تحلیل میان‌مدت
SIGNALS_FILE = "data/signals.json"

# تنظیمات API کوکوین
KUCOIN_BASE_URL = "https://api.kucoin.com"
KUCOIN_KLINE_ENDPOINT = "/api/v1/market/candles"
KUCOIN_TICKER_ENDPOINT = "/api/v1/market/orderbook/level1"
KUCOIN_STATS_ENDPOINT = "/api/v1/market/stats"

# وزن‌دهی فاکتورها برای سیستم امتیازدهی (تمرکز شدید روی استراتژی اصلی)
SIGNAL_WEIGHTS = {
    'rsi': 10,         # تأیید اشباع خرید/فروش (کم اهمیت‌تر)
    'ema': 40,         # وزن بسیار بالا برای کراس EMA (ستون اصلی)
    'macd': 15,        # تأیید مومنتوم (کم اهمیت‌تر از EMA)
    'bb': 0,           # غیرفعال، چون در استراتژی اصلی استفاده نمی‌شود
    'adx': 20,         # تأیید قدرت روند (مهم برای وین ریت)
    'ichi': 0,         # غیرفعال
    'divergence': 0,   # غیرفعال، چون تأیید اضافی پیچیده است
    'candle': 0,       # غیرفعال، الگوهای کندلی در این استراتژی کم اهمیت
    'volume': 5,       # تأیید حجم (کم اهمیت)
    'support': 20,     # نقاط ورود دقیق (مهم)
    'resistance': 20,  # نقاط خروج دقیق (مهم)
    'higher_tf': 15,   # تأیید روند تایم فریم بالاتر
}