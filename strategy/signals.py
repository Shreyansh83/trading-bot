import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from .weights import WeightManager


class WeightedSignalGenerator:
    """
    Generate trading signals using weighted combination of indicators
    """
    
    def __init__(self, weight_manager: Optional[WeightManager] = None):
        self.weight_manager = weight_manager or WeightManager()
    
    def evaluate_conditions(self, df: pd.DataFrame, symbol: str = "default") -> Tuple[Optional[bool], str]:
        """
        Evaluate trading conditions using weighted indicators
        
        Args:
            df: DataFrame with all indicators calculated
            symbol: Trading symbol for weight lookup
            
        Returns:
            Tuple of (signal, reason) where signal is True/False/None for BUY/SELL/NEUTRAL
        """
        if df is None or df.empty:
            return None, "No data"
        
        # Get weights for this symbol
        weights = self.weight_manager.get_weights(symbol)
        
        # Calculate composite score
        composite_score = self._calculate_composite_score(df, weights)
        
        # Generate signal based on composite score
        signal, reason = self._generate_signal_from_score(composite_score, weights, df)
        
        return signal, reason
    
    def _calculate_composite_score(self, df: pd.DataFrame, weights: Dict[str, float]) -> float:
        """
        Calculate weighted composite score from all indicators
        
        Args:
            df: DataFrame with indicators and normalized scores
            weights: Weights dictionary
            
        Returns:
            Composite score between -1 and 1
        """
        if df.empty:
            return 0.0
        
        last_row = df.iloc[-1]
        composite_score = 0.0
        
        # RSI component
        if 'rsi' in last_row and not pd.isna(last_row['rsi']):
            rsi_score = self._normalize_rsi(last_row['rsi'])
            composite_score += rsi_score * weights.get('rsi', 0)
        
        # MACD component
        if 'macd_hist' in last_row and not pd.isna(last_row['macd_hist']):
            macd_score = self._normalize_macd(last_row['macd_hist'])
            composite_score += macd_score * weights.get('macd', 0)
        
        # Bollinger Bands component
        if 'bb_position' in last_row and not pd.isna(last_row['bb_position']):
            bb_score = self._normalize_bb_position(last_row['bb_position'])
            composite_score += bb_score * weights.get('bb', 0)
        
        # EMA crossover component
        if 'ema_crossover' in last_row and not pd.isna(last_row['ema_crossover']):
            ema_score = self._normalize_ema_crossover(last_row, df)
            composite_score += ema_score * weights.get('ema', 0)
        
        # Parabolic SAR component
        if 'sar_signal' in last_row and not pd.isna(last_row['sar_signal']):
            sar_score = last_row['sar_signal']  # Already -1 or 1
            composite_score += sar_score * weights.get('sar', 0)
        
        # SuperTrend component
        if 'supertrend_signal' in last_row and not pd.isna(last_row['supertrend_signal']):
            supertrend_score = last_row['supertrend_signal']  # Already -1 or 1
            composite_score += supertrend_score * weights.get('supertrend', 0)
        
        return np.clip(composite_score, -1, 1)
    
    def _normalize_rsi(self, rsi_value: float) -> float:
        """Normalize RSI to score between -1 and 1"""
        if rsi_value <= 20:
            return 1.0  # Oversold - BUY signal
        elif rsi_value >= 80:
            return -1.0  # Overbought - SELL signal
        else:
            # Linear interpolation between -1 and 1
            return (50 - rsi_value) / 30
    
    def _normalize_macd(self, macd_hist: float) -> float:
        """Normalize MACD histogram to score between -1 and 1"""
        # Simple normalization - can be improved with rolling statistics
        if macd_hist > 0.5:
            return 1.0
        elif macd_hist < -0.5:
            return -1.0
        else:
            return macd_hist * 2  # Scale to -1 to 1
    
    def _normalize_bb_position(self, bb_position: float) -> float:
        """Normalize Bollinger Bands position to score between -1 and 1"""
        if bb_position <= 0.2:
            return 1.0  # Near lower band - BUY signal
        elif bb_position >= 0.8:
            return -1.0  # Near upper band - SELL signal
        else:
            # Convert 0-1 position to -1 to 1 score
            return (0.5 - bb_position) * 2
    
    def _normalize_ema_crossover(self, last_row: pd.Series, df: pd.DataFrame) -> float:
        """Normalize EMA crossover to score between -1 and 1"""
        ema_diff = last_row['ema_crossover']
        close_price = last_row['close']
        
        # Normalize by percentage of close price
        percentage_diff = ema_diff / close_price
        
        # Scale to reasonable range
        return np.clip(percentage_diff * 100, -1, 1)
    
    def _generate_signal_from_score(self, composite_score: float, weights: Dict[str, float], df: pd.DataFrame) -> Tuple[Optional[bool], str]:
        """
        Generate trading signal from composite score
        
        Args:
            composite_score: Weighted composite score
            weights: Weights configuration
            df: DataFrame for additional context
            
        Returns:
            Tuple of (signal, reason)
        """
        buy_threshold = weights.get('buy_threshold', 0.3)
        sell_threshold = weights.get('sell_threshold', -0.3)
        
        if composite_score >= buy_threshold:
            return True, f"BUY signal (score: {composite_score:.3f})"
        elif composite_score <= sell_threshold:
            return False, f"SELL signal (score: {composite_score:.3f})"
        else:
            return None, f"NEUTRAL (score: {composite_score:.3f})"
    
    def get_signal_breakdown(self, df: pd.DataFrame, symbol: str = "default") -> Dict[str, float]:
        """
        Get detailed breakdown of how each indicator contributes to the signal
        
        Args:
            df: DataFrame with indicators
            symbol: Trading symbol
            
        Returns:
            Dictionary with individual indicator contributions
        """
        if df.empty:
            return {}
        
        weights = self.weight_manager.get_weights(symbol)
        last_row = df.iloc[-1]
        breakdown = {}
        
        # Calculate individual contributions
        if 'rsi' in last_row:
            rsi_score = self._normalize_rsi(last_row['rsi'])
            breakdown['rsi_contribution'] = rsi_score * weights.get('rsi', 0)
            breakdown['rsi_score'] = rsi_score
        
        if 'macd_hist' in last_row:
            macd_score = self._normalize_macd(last_row['macd_hist'])
            breakdown['macd_contribution'] = macd_score * weights.get('macd', 0)
            breakdown['macd_score'] = macd_score
        
        if 'bb_position' in last_row:
            bb_score = self._normalize_bb_position(last_row['bb_position'])
            breakdown['bb_contribution'] = bb_score * weights.get('bb', 0)
            breakdown['bb_score'] = bb_score
        
        if 'ema_crossover' in last_row:
            ema_score = self._normalize_ema_crossover(last_row, df)
            breakdown['ema_contribution'] = ema_score * weights.get('ema', 0)
            breakdown['ema_score'] = ema_score
        
        if 'sar_signal' in last_row:
            breakdown['sar_contribution'] = last_row['sar_signal'] * weights.get('sar', 0)
            breakdown['sar_score'] = last_row['sar_signal']
        
        if 'supertrend_signal' in last_row:
            breakdown['supertrend_contribution'] = last_row['supertrend_signal'] * weights.get('supertrend', 0)
            breakdown['supertrend_score'] = last_row['supertrend_signal']
        
        # Total composite score
        breakdown['composite_score'] = sum(v for k, v in breakdown.items() if k.endswith('_contribution'))
        
        return breakdown