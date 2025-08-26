import talib


def calculate_indicators(df):
    """
    Add RSI, MACD, Bollinger Bands and bb_position to DataFrame.
    """
    df = df.copy()
    # RSI-9
    df['rsi'] = talib.RSI(df['close'], timeperiod=9)

    # MACD (5,15,9)
    macd, macd_signal, macd_hist = talib.MACD(
        df['close'], fastperiod=5, slowperiod=15, signalperiod=9
    )
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd_hist

    # Bollinger Bands (10, 2)
    upper, middle, lower = talib.BBANDS(
        df['close'], timeperiod=10, nbdevup=2, nbdevdn=2
    )
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    df['bb_position'] = (df['close'] - lower) / (upper - lower)

    return df