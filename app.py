from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
from datetime import datetime, time
import os
from config import (
    SYMBOL, RESOLUTIONS, DEFAULT_RESOLUTION, NUM_CANDLES, FLASK_SECRET_KEY,
    BASIC_INDICATORS_CONFIG, ADVANCED_INDICATORS_CONFIG
)
from fyers_api import initialize_fyers, get_historical_data, get_historical_range
from indicators.composite import calculate_composite_indicators
from strategy.signals import WeightedSignalGenerator
from strategy.weights import WeightManager
from plotter import save_chart

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# Initialize Fyers and Strategy components
fyers = initialize_fyers()
weight_manager = WeightManager()
signal_generator = WeightedSignalGenerator(weight_manager)

# Ensure static images folder exists
IMG_FOLDER = os.path.join(app.static_folder, 'charts')
os.makedirs(IMG_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles historic POST requests and renders main dashboard
    """
    if request.method == 'POST':
        symbol = request.form.get('symbol') or SYMBOL
        resolution = request.form.get('resolution') or DEFAULT_RESOLUTION
        date_str = request.form.get('historic_date')
        show_chart = request.form.get('show_chart') == 'on'

        if not date_str:
            flash('Please select a date for historic data.')
            return redirect(url_for('index'))

        day = datetime.strptime(date_str, '%Y-%m-%d').date()
        market_open = datetime.combine(day, time(9, 15))
        market_close = datetime.combine(day, time(15, 30))

        # Fetch historical data
        df = get_historical_range(fyers, symbol, resolution, market_open, market_close)
        if df.empty:
            flash(f'No data for {symbol} on {date_str} at {resolution} resolution.')
            return redirect(url_for('index'))

        # Calculate all indicators
        df_ind = calculate_composite_indicators(
            df, 
            basic_config=BASIC_INDICATORS_CONFIG,
            advanced_config=ADVANCED_INDICATORS_CONFIG
        )

        # Generate signals for each row
        signals, reasons, scores = [], [], []
        for i in range(len(df_ind)):
            sig, rsn = signal_generator.evaluate_conditions(df_ind.iloc[:i+1], symbol)
            
            # Get detailed breakdown for last row
            if i == len(df_ind) - 1:
                breakdown = signal_generator.get_signal_breakdown(df_ind.iloc[:i+1], symbol)
                scores.append(breakdown.get('composite_score', 0))
            else:
                scores.append(0)  # Only calculate for last row to save computation
            
            signals.append('BUY' if sig else 'SELL' if sig is False else 'NEUTRAL')
            reasons.append(rsn)

        df_ind['signal'] = signals
        df_ind['reason'] = reasons
        df_ind['composite_score'] = scores

        # Generate chart if requested
        chart_file = None
        if show_chart:
            chart_file = f'charts/historic_{resolution}_{date_str}.png'
            save_chart(df_ind, os.path.join(app.static_folder, chart_file))

        # Prepare data for display
        table_html = df_ind.to_html(classes='table table-striped', index=False)
        
        # Add signal breakdown for the last signal
        signal_breakdown = None
        if not df_ind.empty:
            signal_breakdown = signal_generator.get_signal_breakdown(df_ind, symbol)

        return render_template('historic.html',
                               tables=[table_html],
                               symbol=symbol,
                               date=date_str,
                               resolution=resolution,
                               chart_file=chart_file,
                               signal_breakdown=signal_breakdown)

    # GET request - render main form
    return render_template('index.html',
                           default_symbol=SYMBOL,
                           resolutions=RESOLUTIONS,
                           default_resolution=DEFAULT_RESOLUTION)


@app.route('/live', methods=['GET'])
def live_view():
    """
    Handles live data view with enhanced signal generation
    """
    symbol = request.args.get('symbol') or SYMBOL
    resolution = request.args.get('resolution') or DEFAULT_RESOLUTION
    show_chart = request.args.get('show_chart') == 'on'

    # Fetch recent data
    df = get_historical_data(fyers, symbol, resolution, NUM_CANDLES)
    if df.empty:
        flash(f'No live data for {symbol} at {resolution} resolution.')
        return redirect(url_for('index'))

    # Calculate all indicators
    df_ind = calculate_composite_indicators(
        df,
        basic_config=BASIC_INDICATORS_CONFIG,
        advanced_config=ADVANCED_INDICATORS_CONFIG
    )

    # Generate signal
    signal, reason = signal_generator.evaluate_conditions(df_ind, symbol)
    signal_breakdown = signal_generator.get_signal_breakdown(df_ind, symbol)
    
    price = df['close'].iloc[-1]
    last_update = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Prepare table with indicator values
    display_columns = [
        'timestamp', 'close', 'rsi', 'macd_hist', 'bb_position', 
        'ema_crossover', 'sar_signal', 'supertrend_signal'
    ]
    available_columns = [col for col in display_columns if col in df_ind.columns]
    table_html = df_ind[available_columns].tail(10).to_html(classes='table table-sm', index=False)

    # Generate chart if requested
    chart_file = None
    if show_chart:
        ts = datetime.now().strftime("%H%M%S")
        chart_file = f'charts/live_{resolution}_{ts}.png'
        save_chart(df_ind, os.path.join(app.static_folder, chart_file))

    return render_template('live.html',
                           symbol=symbol,
                           resolution=resolution,
                           price=price,
                           signal=signal,
                           reason=reason,
                           last_update=last_update,
                           table_html=table_html,
                           chart_file=chart_file,
                           signal_breakdown=signal_breakdown)


@app.route('/weights', methods=['GET', 'POST'])
def weights_management():
    """
    Manage indicator weights for different symbols
    """
    if request.method == 'POST':
        symbol = request.form.get('symbol', 'default')
        
        # Get weights from form
        weights = {}
        for indicator in ['rsi', 'macd', 'bb', 'ema', 'sar', 'supertrend']:
            weight_value = request.form.get(f'weight_{indicator}')
            if weight_value:
                weights[indicator] = float(weight_value)
        
        # Get thresholds
        buy_threshold = request.form.get('buy_threshold')
        sell_threshold = request.form.get('sell_threshold')
        
        if buy_threshold:
            weights['buy_threshold'] = float(buy_threshold)
        if sell_threshold:
            weights['sell_threshold'] = float(sell_threshold)
        
        # Normalize and save weights
        normalized_weights = weight_manager.normalize_weights(weights)
        normalized_weights['buy_threshold'] = weights.get('buy_threshold', 0.3)
        normalized_weights['sell_threshold'] = weights.get('sell_threshold', -0.3)
        
        weight_manager.set_weights(normalized_weights, symbol)
        flash(f'Weights updated for {symbol}')
        
        return redirect(url_for('weights_management'))
    
    # GET request - show weights management page
    symbol = request.args.get('symbol', 'default')
    current_weights = weight_manager.get_weights(symbol)
    
    return render_template('weights.html',
                           symbol=symbol,
                           weights=current_weights,
                           available_symbols=['default', 'NSE:RELIANCE-EQ', 'NSE:TCS-EQ'])


if __name__ == '__main__':
    app.run(debug=True)