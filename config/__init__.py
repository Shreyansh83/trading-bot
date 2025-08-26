"""
Configuration Module

This module provides centralized configuration management
for the trading bot application.
"""

from .trading_params import *

# Backward compatibility - expose all trading params at package level
__all__ = [
    'CLIENT_ID',
    'SECRET_KEY', 
    'REDIRECT_URI',
    'AUTH_CODE',
    'SYMBOL',
    'DEFAULT_RESOLUTION',
    'NUM_CANDLES',
    'RESOLUTIONS',
    'CHECK_INTERVAL_SECONDS',
    'DURATION_MINUTES',
    'BASIC_INDICATORS_CONFIG',
    'ADVANCED_INDICATORS_CONFIG',
    'DEFAULT_WEIGHTS',
    'FLASK_SECRET_KEY'
]