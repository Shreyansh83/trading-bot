"""
Trading Parameters Configuration

This module contains all trading-related configuration parameters
that were previously in config.py
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fyers API credentials
CLIENT_ID = os.getenv("CLIENT_ID")
SECRET_KEY = os.getenv("SECRET_KEY")
REDIRECT_URI = os.getenv("REDIRECT_URI")
AUTH_CODE = os.getenv("AUTH_CODE", "")

# Trading Parameters
SYMBOL = os.getenv("SYMBOL", "NSE:RELIANCE-EQ")
DEFAULT_RESOLUTION = os.getenv("DEFAULT_RESOLUTION", "5")    # 5-minute
NUM_CANDLES = int(os.getenv("NUM_CANDLES", "15"))            # Default bars

# Supported resolutions (minutes) for front-end
RESOLUTIONS = [
    ("1", "1 Minute"),
    ("5", "5 Minutes"),
    ("30", "30 Minutes"),
    ("60", "1 Hour"),
    ("120", "2 Hours"),
    ("D", "Daily")
]

# Live loop timing
CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))  # 5 minutes
DURATION_MINUTES = int(os.getenv("DURATION_MINUTES", "60"))              # default run duration

# Basic Indicator parameters
BASIC_INDICATORS_CONFIG = {
    'rsi_period': int(os.getenv("RSI_PERIOD", "9")),
    'macd_fast': int(os.getenv("MACD_FAST_PERIOD", "5")),
    'macd_slow': int(os.getenv("MACD_SLOW_PERIOD", "15")),
    'macd_signal': int(os.getenv("MACD_SIGNAL_PERIOD", "9")),
    'bb_period': int(os.getenv("BB_PERIOD", "10")),
    'bb_stddev': int(os.getenv("BB_STDDEV", "2"))
}

# Advanced Indicator parameters
ADVANCED_INDICATORS_CONFIG = {
    'ema_short': int(os.getenv("EMA_SHORT_PERIOD", "9")),
    'ema_long': int(os.getenv("EMA_LONG_PERIOD", "21")),
    'sar_acceleration': float(os.getenv("SAR_ACCELERATION", "0.02")),
    'sar_maximum': float(os.getenv("SAR_MAXIMUM", "0.2")),
    'supertrend_period': int(os.getenv("SUPERTREND_PERIOD", "10")),
    'supertrend_multiplier': float(os.getenv("SUPERTREND_MULTIPLIER", "3.0"))
}

# Default weight configuration (will be moved to WeightManager)
DEFAULT_WEIGHTS = {
    'rsi': float(os.getenv("WEIGHT_RSI", "0.15")),
    'macd': float(os.getenv("WEIGHT_MACD", "0.20")),
    'bb': float(os.getenv("WEIGHT_BB", "0.15")),
    'ema': float(os.getenv("WEIGHT_EMA", "0.20")),
    'sar': float(os.getenv("WEIGHT_SAR", "0.15")),
    'supertrend': float(os.getenv("WEIGHT_SUPERTREND", "0.15")),
    'buy_threshold': float(os.getenv("BUY_THRESHOLD", "0.3")),
    'sell_threshold': float(os.getenv("SELL_THRESHOLD", "-0.3"))
}

# Flask app settings
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "replace_with_secure_random_string")