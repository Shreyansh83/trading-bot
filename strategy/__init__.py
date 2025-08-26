"""
Strategy Module

This module handles trading strategy logic including:
- Weight management
- Signal generation
- Backtesting functionality
"""

from .weights import WeightManager
from .signals import WeightedSignalGenerator
from .backtesting import StrategyBacktester

__all__ = [
    'WeightManager',
    'WeightedSignalGenerator', 
    'StrategyBacktester'
]