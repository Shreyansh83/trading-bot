from config import (
    RSI_UPPER_THRESHOLD, RSI_LOWER_THRESHOLD,
    MACD_SIGNAL_DIFF_THRESHOLD, BB_POSITION_THRESHOLD
)


def evaluate_conditions(df):
    """
    Return (signal: bool|None, reason: str).
    True = BUY, False = SELL/AVOID, None = NEUTRAL
    """
    if df is None or df.empty:
        return None, "No data"
    last = df.iloc[-1]
    rsi = last['rsi']
    macd_diff = last['macd'] - last['macd_signal']
    bb_pos = last['bb_position']
    # EMA
    # PARABOLIC SAR
    # SUPER TREND
    # RESOLUTION 1 MINUTE

    # RSI conditions
    if rsi <= RSI_LOWER_THRESHOLD:
        return True, f"RSI oversold ({rsi:.2f})"
    if rsi >= RSI_UPPER_THRESHOLD:
        return False, f"RSI overbought ({rsi:.2f})"

    # MACD crossover
    if macd_diff > MACD_SIGNAL_DIFF_THRESHOLD:
        return True, f"MACD bullish ({macd_diff:.2f})"
    if macd_diff < -MACD_SIGNAL_DIFF_THRESHOLD:
        return False, f"MACD bearish ({macd_diff:.2f})"

    # Bollinger Bands
    if bb_pos < (1 - BB_POSITION_THRESHOLD):
        return True, f"Near lower BB ({bb_pos:.2f})"
    if bb_pos > BB_POSITION_THRESHOLD:
        return False, f"Near upper BB ({bb_pos:.2f})"

    # No clear signal
    return None, "No strong signal"