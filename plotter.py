import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

plt.ioff()  # Turn off interactive mode for file saving

def save_chart(df, filename):
    """
    Enhanced chart with all indicators
    """
    # Prepare figure with more subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    ax_price, ax_rsi, ax_macd, ax_ema = axes
    
    # Price chart with Bollinger Bands and SuperTrend
    ax_price.plot(df['timestamp'], df['close'], label='Close', linewidth=1.5)
    
    if 'bb_upper' in df.columns:
        ax_price.plot(df['timestamp'], df['bb_upper'], 'r--', alpha=0.7, label='BB Upper')
        ax_price.plot(df['timestamp'], df['bb_lower'], 'g--', alpha=0.7, label='BB Lower')
        ax_price.fill_between(df['timestamp'], df['bb_upper'], df['bb_lower'], alpha=0.1)
    
    if 'supertrend' in df.columns:
        # Color SuperTrend based on signal
        for i in range(1, len(df)):
            if df['supertrend_signal'].iloc[i] == 1:
                ax_price.plot(df['timestamp'].iloc[i-1:i+1], df['supertrend'].iloc[i-1:i+1], 'g-', linewidth=2)
            else:
                ax_price.plot(df['timestamp'].iloc[i-1:i+1], df['supertrend'].iloc[i-1:i+1], 'r-', linewidth=2)
    
    # Add Parabolic SAR
    if 'sar' in df.columns:
        bullish_sar = df[df['sar_signal'] == 1]['sar']
        bearish_sar = df[df['sar_signal'] == -1]['sar']
        bullish_times = df[df['sar_signal'] == 1]['timestamp']
        bearish_times = df[df['sar_signal'] == -1]['timestamp']
        
        ax_price.scatter(bullish_times, bullish_sar, color='green', marker='^', s=20, alpha=0.7, label='SAR Bullish')
        ax_price.scatter(bearish_times, bearish_sar, color='red', marker='v', s=20, alpha=0.7, label='SAR Bearish')
    
    ax_price.set_ylabel('Price')
    ax_price.legend(loc='upper left')
    ax_price.grid(True, alpha=0.3)
    
    # RSI with overbought/oversold levels
    if 'rsi' in df.columns:
        ax_rsi.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
        ax_rsi.axhline(80, linestyle='--', color='red', alpha=0.7)
        ax_rsi.axhline(20, linestyle='--', color='green', alpha=0.7)
        ax_rsi.fill_between(df['timestamp'], 20, 80, alpha=0.1)
        ax_rsi.set_ylabel('RSI')
        ax_rsi.set_ylim(0, 100)
        ax_rsi.legend()
        ax_rsi.grid(True, alpha=0.3)
    
    # MACD
    if 'macd' in df.columns:
        ax_macd.plot(df['timestamp'], df['macd'], label='MACD', color='blue')
        ax_macd.plot(df['timestamp'], df['macd_signal'], label='Signal', color='red')
        ax_macd.bar(df['timestamp'], df['macd_hist'], label='Histogram', alpha=0.7, width=0.8)
        ax_macd.axhline(0, linestyle='--', color='grey', alpha=0.7)
        ax_macd.set_ylabel('MACD')
        ax_macd.legend()
        ax_macd.grid(True, alpha=0.3)
    
    # EMA and other indicators
    if 'ema_short' in df.columns and 'ema_long' in df.columns:
        ax_ema.plot(df['timestamp'], df['ema_short'], label='EMA Short', color='blue')
        ax_ema.plot(df['timestamp'], df['ema_long'], label='EMA Long', color='red')
        
        # Fill area between EMAs
        ax_ema.fill_between(df['timestamp'], df['ema_short'], df['ema_long'], 
                           where=(df['ema_short'] >= df['ema_long']), color='green', alpha=0.3, label='Bullish')
        ax_ema.fill_between(df['timestamp'], df['ema_short'], df['ema_long'], 
                           where=(df['ema_short'] < df['ema_long']), color='red', alpha=0.3, label='Bearish')
        
        ax_ema.set_ylabel('EMA')
        ax_ema.legend()
        ax_ema.grid(True, alpha=0.3)
    
    # Format x-axis
    ax_ema.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_ema.set_xlabel('Time')
    
    # Rotate x-axis labels for better readability
    plt.setp(ax_ema.xaxis.get_majorticklabels(), rotation=45)
    
    # Add signal markers on price chart
    if 'signal' in df.columns:
        buy_signals = df[df['signal'] == 'BUY']
        sell_signals = df[df['signal'] == 'SELL']
        
        ax_price.scatter(buy_signals['timestamp'], buy_signals['close'], 
                        color='green', marker='^', s=100, zorder=5, label='BUY Signal')
        ax_price.scatter(sell_signals['timestamp'], sell_signals['close'], 
                        color='red', marker='v', s=100, zorder=5, label='SELL Signal')
    
    # Add title
    symbol = getattr(df, 'symbol', 'Unknown')
    fig.suptitle(f'Trading Analysis - {symbol}', fontsize=14, fontweight='bold')
    
    # Adjust layout and save
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Explicitly close the figure
    plt.clf()       # Clear the current figure
    plt.cla()       # Clear the current axes