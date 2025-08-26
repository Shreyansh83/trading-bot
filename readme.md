# Trading Bot

A Flask-based web application for technical analysis and automated trading signals using Fyers API. The bot analyzes stock/instrument data with indicators like RSI, MACD, and Bollinger Bands to generate BUY/SELL signals.

## Features

- **Live Trading Signals**: Get real-time BUY/SELL/NEUTRAL signals with the underlying reason
- **Historical Analysis**: Analyze past market data for backtesting and strategy refinement
- **Technical Indicators**: Uses RSI, MACD, and Bollinger Bands for signal generation
- **Interactive Dashboard**: Simple web interface for parameter configuration
- **Chart Visualization**: Visual representation of price movements and indicators

## Technical Stack

- **Backend**: Flask (Python)
- **Data Analysis**: Pandas, TA-Lib
- **API Integration**: Fyers API v3
- **Visualization**: Matplotlib
- **Frontend**: Bootstrap 5

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Install TA-Lib (Technical Analysis Library):
   - For Windows: Download and install from [TA-Lib binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
   - For macOS: `brew install ta-lib`
   - For Linux: `sudo apt-get install ta-lib`

4. Create a `.env` file in the project root directory (see Configuration section)

## Configuration

Create a `.env` file with the following structure and add your Fyers API credentials:

```
# Fyers API credentials
CLIENT_ID=YOUR_CLIENT_ID
SECRET_KEY=YOUR_SECRET_KEY
REDIRECT_URI=YOUR_REDIRECT_URI
AUTH_CODE=YOUR_AUTHENTICATION_CODE

# Trading Parameters
SYMBOL=NSE:RELIANCE-EQ
DEFAULT_RESOLUTION=5
NUM_CANDLES=15

# Signal thresholds
RSI_UPPER_THRESHOLD=80
RSI_LOWER_THRESHOLD=20
MACD_SIGNAL_DIFF_THRESHOLD=0.3
BB_POSITION_THRESHOLD=0.8

# Flask app settings
FLASK_SECRET_KEY=your_secure_random_string
```

You can customize any of these parameters to fit your trading strategy.

## Usage

1. Start the application:

```bash
python app.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. The dashboard provides two main modes:
   - **Historic**: Analyze past data for a specific date
   - **Live**: Monitor real-time signals with auto-refresh

## Trading Strategy

The bot uses a combination of technical indicators to generate signals:

1. **RSI (Relative Strength Index)**:
   - BUY when RSI ≤ 20 (oversold)
   - SELL when RSI ≥ 80 (overbought)

2. **MACD (Moving Average Convergence Divergence)**:
   - BUY when MACD line crosses above signal line by threshold
   - SELL when MACD line crosses below signal line by threshold

3. **Bollinger Bands**:
   - BUY when price is near lower band
   - SELL when price is near upper band

## Project Structure

- `app.py`: Main Flask application
- `config.py`: Configuration parameters (loads from .env)
- `fyers_api.py`: Fyers API integration
- `indicators.py`: Technical indicator calculations
- `strategy.py`: Trading signal logic
- `plotter.py`: Chart generation
- `.env`: Environment variables for configuration (not committed to Git)
- `.gitignore`: Files and directories to exclude from Git
- `templates/`: HTML templates
  - `index.html`: Dashboard form
  - `live.html`: Live signal view
  - `historic.html`: Historical analysis view

## Security Notes

- The `.env` file containing your API credentials is added to `.gitignore` to prevent it from being committed to version control
- Use a strong, random string for the `FLASK_SECRET_KEY` variable
- Consider using environment variables in production environments

## Disclaimer

This trading bot is for educational and research purposes only. Always perform your own analysis before making trading decisions. Past performance is not indicative of future results.

## License

MIT