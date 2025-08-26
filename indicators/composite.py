import pandas as pd
from .basic import calculate_basic_indicators
from .advanced import calculate_advanced_indicators


def calculate_composite_indicators(df, basic_config=None, advanced_config=None):
    """
    Calculate all indicators (basic + advanced) in one function
    
    Args:
        df: DataFrame with OHLCV data
        basic_config: Configuration for basic indicators
        advanced_config: Configuration for advanced indicators
        
    Returns:
        DataFrame with all indicators added
    """
    # Calculate basic indicators
    df = calculate_basic_indicators(df, basic_config)
    
    # Calculate advanced indicators
    df = calculate_advanced_indicators(df, advanced_config)
    
    return df


def calculate_normalized_scores(df, normalization_config=None):
    """
    Normalize indicator values to scores between -1 and 1
    
    Args:
        df: DataFrame with all indicators calculated
        normalization_config: Configuration for score normalization
        
    Returns:
        DataFrame with normalized scores added
    """
    if normalization_config is None:
        normalization_config = {
            'rsi_overbought': 80,
            'rsi_oversold': 20,
            'bb_upper_threshold': 0.8,
            'bb_lower_threshold': 0.2,
            'macd_threshold': 0.5
        }
    
    df = df.copy()
    
    # RSI Score (-1 to 1)
    df['rsi_score'] = (df['rsi'] - 50) / 50  # Normalize around 50
    
    # MACD Score
    df['macd_score'] = df['macd_hist'] / df['close'].rolling(20).std()  # Normalize by price volatility
    df['macd_score'] = df['macd_score'].clip(-1, 1)
    
    # Bollinger Bands Score
    df['bb_score'] = (df['bb_position'] - 0.5) * 2  # Convert 0-1 to -1-1
    
    # EMA Crossover Score
    ema_diff = df['ema_crossover'] / df['close']
    df['ema_score'] = (ema_diff / ema_diff.rolling(20).std()).clip(-1, 1)
    
    # Parabolic SAR Score (discrete: -1 or 1)
    df['sar_score'] = df['sar_signal']
    
    # SuperTrend Score (discrete: -1 or 1)
    df['supertrend_score'] = df['supertrend_signal']
    
    return df