import talib
import pandas as pd


def calculate_basic_indicators(df, config=None):
    """
    Calculate basic technical indicators: RSI, MACD, Bollinger Bands
    
    Args:
        df: DataFrame with OHLCV data
        config: Dictionary with indicator parameters
        
    Returns:
        DataFrame with basic indicators added
    """
    if config is None:
        config = {
            'rsi_period': 9,
            'macd_fast': 5,
            'macd_slow': 15,
            'macd_signal': 9,
            'bb_period': 10,
            'bb_stddev': 2
        }
    
    df = df.copy()
    
    # RSI
    df['rsi'] = talib.RSI(df['close'], timeperiod=config['rsi_period'])
    
    # MACD
    macd, macd_signal, macd_hist = talib.MACD(
        df['close'], 
        fastperiod=config['macd_fast'], 
        slowperiod=config['macd_slow'], 
        signalperiod=config['macd_signal']
    )
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist
    
    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(
        df['close'], 
        timeperiod=config['bb_period'], 
        nbdevup=config['bb_stddev'], 
        nbdevdn=config['bb_stddev']
    )
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_position'] = (df['close'] - lower) / (upper - lower)
    
    return df