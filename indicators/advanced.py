import talib
import numpy as np
import pandas as pd


def calculate_advanced_indicators(df, config=None):
    """
    Calculate advanced technical indicators: EMA, Parabolic SAR, SuperTrend
    
    Args:
        df: DataFrame with OHLCV data
        config: Dictionary with indicator parameters
        
    Returns:
        DataFrame with advanced indicators added
    """
    if config is None:
        config = {
            'ema_short': 9,
            'ema_long': 21,
            'sar_acceleration': 0.02,
            'sar_maximum': 0.2,
            'supertrend_period': 10,
            'supertrend_multiplier': 3.0
        }
    
    df = df.copy()
    
    # EMA (Exponential Moving Averages)
    df['ema_short'] = talib.EMA(df['close'], timeperiod=config['ema_short'])
    df['ema_long'] = talib.EMA(df['close'], timeperiod=config['ema_long'])
    df['ema_crossover'] = df['ema_short'] - df['ema_long']
    
    # Parabolic SAR
    df['sar'] = talib.SAR(
        df['high'], 
        df['low'],
        acceleration=config['sar_acceleration'],
        maximum=config['sar_maximum']
    )
    df['sar_signal'] = np.where(df['close'] > df['sar'], 1, -1)
    
    # SuperTrend
    df = calculate_supertrend(
        df, 
        period=config['supertrend_period'],
        multiplier=config['supertrend_multiplier']
    )
    
    return df


def calculate_supertrend(df, period=10, multiplier=3.0):
    """
    Calculate SuperTrend indicator
    
    Args:
        df: DataFrame with OHLCV data
        period: Period for ATR calculation
        multiplier: Multiplier for ATR
        
    Returns:
        DataFrame with SuperTrend columns added
    """
    df = df.copy()
    
    # Calculate ATR (Average True Range)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
    
    # Calculate basic upper and lower bands
    hl2 = (df['high'] + df['low']) / 2
    df['basic_upper'] = hl2 + (multiplier * df['atr'])
    df['basic_lower'] = hl2 - (multiplier * df['atr'])
    
    # Calculate final upper and lower bands
    df['final_upper'] = 0.0
    df['final_lower'] = 0.0
    df['supertrend'] = 0.0
    df['supertrend_signal'] = 0
    
    for i in range(1, len(df)):
        # Final Upper Band
        if (df['basic_upper'].iloc[i] < df['final_upper'].iloc[i-1] or 
            df['close'].iloc[i-1] > df['final_upper'].iloc[i-1]):
            df.iloc[i, df.columns.get_loc('final_upper')] = df['basic_upper'].iloc[i]
        else:
            df.iloc[i, df.columns.get_loc('final_upper')] = df['final_upper'].iloc[i-1]
        
        # Final Lower Band
        if (df['basic_lower'].iloc[i] > df['final_lower'].iloc[i-1] or 
            df['close'].iloc[i-1] < df['final_lower'].iloc[i-1]):
            df.iloc[i, df.columns.get_loc('final_lower')] = df['basic_lower'].iloc[i]
        else:
            df.iloc[i, df.columns.get_loc('final_lower')] = df['final_lower'].iloc[i-1]
        
        # SuperTrend
        if (df['supertrend'].iloc[i-1] == df['final_upper'].iloc[i-1] and 
            df['close'].iloc[i] <= df['final_upper'].iloc[i]):
            df.iloc[i, df.columns.get_loc('supertrend')] = df['final_upper'].iloc[i]
        elif (df['supertrend'].iloc[i-1] == df['final_upper'].iloc[i-1] and 
              df['close'].iloc[i] > df['final_upper'].iloc[i]):
            df.iloc[i, df.columns.get_loc('supertrend')] = df['final_lower'].iloc[i]
        elif (df['supertrend'].iloc[i-1] == df['final_lower'].iloc[i-1] and 
              df['close'].iloc[i] >= df['final_lower'].iloc[i]):
            df.iloc[i, df.columns.get_loc('supertrend')] = df['final_lower'].iloc[i]
        elif (df['supertrend'].iloc[i-1] == df['final_lower'].iloc[i-1] and 
              df['close'].iloc[i] < df['final_lower'].iloc[i]):
            df.iloc[i, df.columns.get_loc('supertrend')] = df['final_upper'].iloc[i]
    
    # SuperTrend Signal (1 for bullish, -1 for bearish)
    df['supertrend_signal'] = np.where(df['close'] > df['supertrend'], 1, -1)
    
    # Clean up intermediate columns (optional)
    df.drop(['basic_upper', 'basic_lower', 'final_upper', 'final_lower'], axis=1, inplace=True)
    
    return df