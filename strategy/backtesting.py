import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


class StrategyBacktester:
    """
    Backtest trading strategies with various performance metrics
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.commission = 0.001  # 0.1% commission per trade
    
    def backtest_strategy(self, df: pd.DataFrame, signals: List[str], prices: List[float]) -> Dict:
        """
        Backtest a strategy given signals and prices
        
        Args:
            df: DataFrame with price and indicator data
            signals: List of signals ('BUY', 'SELL', 'NEUTRAL')
            prices: List of prices corresponding to signals
            
        Returns:
            Dictionary with performance metrics
        """
        if len(signals) != len(prices):
            raise ValueError("Signals and prices must have same length")
        
        # Initialize tracking variables
        portfolio = {
            'cash': self.initial_capital,
            'position': 0,  # Number of shares
            'entry_price': 0,
            'trades': [],
            'equity_curve': []
        }
        
        # Execute trades based on signals
        for i, (signal, price) in enumerate(zip(signals, prices)):
            timestamp = df.index[i] if hasattr(df, 'index') else i
            
            if signal == 'BUY' and portfolio['position'] == 0:
                # Enter long position
                shares_to_buy = int(portfolio['cash'] / (price * (1 + self.commission)))
                if shares_to_buy > 0:
                    cost = shares_to_buy * price * (1 + self.commission)
                    portfolio['cash'] -= cost
                    portfolio['position'] = shares_to_buy
                    portfolio['entry_price'] = price
                    
                    portfolio['trades'].append({
                        'timestamp': timestamp,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares_to_buy,
                        'value': cost
                    })
            
            elif signal == 'SELL' and portfolio['position'] > 0:
                # Exit long position
                sale_value = portfolio['position'] * price * (1 - self.commission)
                portfolio['cash'] += sale_value
                
                # Calculate trade return
                trade_return = (price - portfolio['entry_price']) / portfolio['entry_price']
                
                portfolio['trades'].append({
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'price': price,
                    'shares': portfolio['position'],
                    'value': sale_value,
                    'return': trade_return
                })
                
                portfolio['position'] = 0
                portfolio['entry_price'] = 0
            
            # Calculate current portfolio value
            current_value = portfolio['cash']
            if portfolio['position'] > 0:
                current_value += portfolio['position'] * price
            
            portfolio['equity_curve'].append({
                'timestamp': timestamp,
                'portfolio_value': current_value,
                'cash': portfolio['cash'],
                'position_value': portfolio['position'] * price if portfolio['position'] > 0 else 0
            })
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(portfolio, df)
        
        return {
            'portfolio': portfolio,
            'performance': performance
        }
    
    def _calculate_performance_metrics(self, portfolio: Dict, df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            portfolio: Portfolio data from backtest
            df: Original price data
            
        Returns:
            Dictionary with performance metrics
        """
        equity_curve = portfolio['equity_curve']
        trades = portfolio['trades']
        
        if not equity_curve:
            return {}
        
        # Basic metrics
        final_value = equity_curve[-1]['portfolio_value']
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate daily returns
        portfolio_values = [eq['portfolio_value'] for eq in equity_curve]
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Risk metrics
        volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = (total_return - 0.02) / volatility if volatility > 0 else 0  # Assuming 2% risk-free rate
        
        # Drawdown calculation
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Trade analysis
        winning_trades = [t for t in trades if t.get('return', 0) > 0]
        losing_trades = [t for t in trades if t.get('return', 0) < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean([t['return'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['return'] for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        total_wins = sum(t['return'] for t in winning_trades)
        total_losses = abs(sum(t['return'] for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annual_return': total_return * (252 / len(equity_curve)) if equity_curve else 0,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_portfolio_value': final_value
        }
    
    def compare_strategies(self, results: List[Dict]) -> pd.DataFrame:
        """
        Compare multiple strategy results
        
        Args:
            results: List of backtest results
            
        Returns:
            DataFrame comparing strategy performance
        """
        comparison_data = []
        
        for i, result in enumerate(results):
            performance = result['performance']
            comparison_data.append({
                'Strategy': f'Strategy_{i+1}',
                'Total Return': f"{performance.get('total_return', 0):.2%}",
                'Annual Return': f"{performance.get('annual_return', 0):.2%}",
                'Volatility': f"{performance.get('volatility', 0):.2%}",
                'Sharpe Ratio': f"{performance.get('sharpe_ratio', 0):.3f}",
                'Max Drawdown': f"{performance.get('max_drawdown', 0):.2%}",
                'Win Rate': f"{performance.get('win_rate', 0):.2%}",
                'Profit Factor': f"{performance.get('profit_factor', 0):.2f}",
                'Total Trades': performance.get('total_trades', 0)
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_performance(self, result: Dict, symbol: str = "Strategy", save_path: Optional[str] = None):
        """
        Plot comprehensive performance analysis
        
        Args:
            result: Backtest result dictionary
            symbol: Symbol or strategy name for the title
            save_path: Optional path to save the plot
        """
        portfolio = result['portfolio']
        performance = result['performance']
        equity_curve = portfolio['equity_curve']
        trades = portfolio['trades']
        
        if not equity_curve:
            print("No data to plot")
            return
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Equity Curve
        timestamps = [eq['timestamp'] for eq in equity_curve]
        portfolio_values = [eq['portfolio_value'] for eq in equity_curve]
        
        ax1.plot(timestamps, portfolio_values, linewidth=2)
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5)
        ax1.set_title(f'Equity Curve - {symbol}')
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True, alpha=0.3)
        
        # Add buy/sell markers
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']
        
        if buy_trades:
            buy_timestamps = [t['timestamp'] for t in buy_trades]
            buy_values = [eq['portfolio_value'] for eq in equity_curve 
                         if eq['timestamp'] in buy_timestamps]
            ax1.scatter(buy_timestamps, buy_values, color='green', marker='^', s=50, zorder=5)
        
        if sell_trades:
            sell_timestamps = [t['timestamp'] for t in sell_trades]
            sell_values = [eq['portfolio_value'] for eq in equity_curve 
                          if eq['timestamp'] in sell_timestamps]
            ax1.scatter(sell_timestamps, sell_values, color='red', marker='v', s=50, zorder=5)
        
        # 2. Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak * 100
        
        ax2.fill_between(timestamps, drawdown, 0, alpha=0.3, color='red')
        ax2.plot(timestamps, drawdown, color='red', linewidth=1)
        ax2.set_title('Drawdown (%)')
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        
        # 3. Trade Returns Distribution
        if trades:
            returns = [t.get('return', 0) * 100 for t in trades if 'return' in t]
            if returns:
                ax3.hist(returns, bins=20, alpha=0.7, edgecolor='black')
                ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                ax3.set_title('Trade Returns Distribution')
                ax3.set_xlabel('Return %')
                ax3.set_ylabel('Frequency')
                ax3.grid(True, alpha=0.3)
        
        # 4. Performance Metrics Table
        ax4.axis('off')
        metrics_text = f"""
        Performance Metrics:
        
        Total Return: {performance.get('total_return', 0):.2%}
        Annual Return: {performance.get('annual_return', 0):.2%}
        Volatility: {performance.get('volatility', 0):.2%}
        Sharpe Ratio: {performance.get('sharpe_ratio', 0):.3f}
        Max Drawdown: {performance.get('max_drawdown', 0):.2%}
        
        Trade Statistics:
        Total Trades: {performance.get('total_trades', 0)}
        Win Rate: {performance.get('win_rate', 0):.2%}
        Avg Win: {performance.get('avg_win', 0):.2%}
        Avg Loss: {performance.get('avg_loss', 0):.2%}
        Profit Factor: {performance.get('profit_factor', 0):.2f}
        
        Final Portfolio Value: ${performance.get('final_portfolio_value', 0):,.2f}
        """
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def backtest_weight_combinations(self, df: pd.DataFrame, prices: List[float], 
                                   weight_combinations: List[Dict]) -> List[Dict]:
        """
        Backtest multiple weight combinations
        
        Args:
            df: DataFrame with price and indicator data
            prices: List of prices
            weight_combinations: List of weight dictionaries to test
            
        Returns:
            List of backtest results for each weight combination
        """
        # Import here to avoid circular import
        from .signals import WeightedSignalGenerator
        from .weights import WeightManager
        
        results = []
        
        for i, weights in enumerate(weight_combinations):
            # Create temporary weight manager with these weights
            weight_manager = WeightManager()
            weight_manager.set_weights(weights, f"test_strategy_{i}")
            
            # Generate signals
            signal_generator = WeightedSignalGenerator(weight_manager)
            signals = []
            
            for j in range(len(df)):
                signal, _ = signal_generator.evaluate_conditions(df.iloc[:j+1], f"test_strategy_{i}")
                signals.append('BUY' if signal else 'SELL' if signal is False else 'NEUTRAL')
            
            # Backtest this strategy
            result = self.backtest_strategy(df, signals, prices)
            result['weights'] = weights
            result['strategy_name'] = f"Strategy_{i+1}"
            results.append(result)
        
        return results


class WeightOptimizer:
    """
    Optimize indicator weights using various methods
    This will be enhanced in Phase 2 with ML algorithms
    """
    
    def __init__(self, backtester: StrategyBacktester):
        self.backtester = backtester
    
    def grid_search_optimization(self, df: pd.DataFrame, prices: List[float],
                                weight_ranges: Dict, step_size: float = 0.05) -> Dict:
        """
        Simple grid search optimization for weights
        
        Args:
            df: DataFrame with price and indicator data
            prices: List of prices
            weight_ranges: Dict with ranges for each indicator
            step_size: Step size for grid search
            
        Returns:
            Best weights and results
        """
        # Generate all weight combinations
        import itertools
        
        weight_combinations = []
        ranges = {}
        
        for indicator, (min_val, max_val) in weight_ranges.items():
            ranges[indicator] = np.arange(min_val, max_val + step_size, step_size)
        
        # Generate all combinations (this will be large, use with caution)
        for combination in itertools.product(*ranges.values()):
            weights = dict(zip(weight_ranges.keys(), combination))
            
            # Normalize weights
            total = sum(weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
                weights['buy_threshold'] = 0.3
                weights['sell_threshold'] = -0.3
                weight_combinations.append(weights)
        
        # Limit combinations to prevent excessive computation
        if len(weight_combinations) > 1000:
            print(f"Too many combinations ({len(weight_combinations)}). Sampling 1000 random combinations.")
            import random
            weight_combinations = random.sample(weight_combinations, 1000)
        
        # Backtest all combinations
        results = self.backtester.backtest_weight_combinations(df, prices, weight_combinations)
        
        # Find best performing strategy (by Sharpe ratio)
        best_result = max(results, key=lambda x: x['performance'].get('sharpe_ratio', -np.inf))
        
        return {
            'best_weights': best_result['weights'],
            'best_result': best_result,
            'all_results': results
        }
    
    def random_search_optimization(self, df: pd.DataFrame, prices: List[float],
                                 n_iterations: int = 100) -> Dict:
        """
        Random search optimization for weights
        
        Args:
            df: DataFrame with price and indicator data
            prices: List of prices
            n_iterations: Number of random weight combinations to try
            
        Returns:
            Best weights and results
        """
        import random
        
        indicators = ['rsi', 'macd', 'bb', 'ema', 'sar', 'supertrend']
        weight_combinations = []
        
        for _ in range(n_iterations):
            # Generate random weights
            weights = {indicator: random.uniform(0, 1) for indicator in indicators}
            
            # Normalize weights
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            
            # Add thresholds
            weights['buy_threshold'] = random.uniform(0.1, 0.5)
            weights['sell_threshold'] = random.uniform(-0.5, -0.1)
            
            weight_combinations.append(weights)
        
        # Backtest all combinations
        results = self.backtester.backtest_weight_combinations(df, prices, weight_combinations)
        
        # Find best performing strategy
        best_result = max(results, key=lambda x: x['performance'].get('sharpe_ratio', -np.inf))
        
        return {
            'best_weights': best_result['weights'],
            'best_result': best_result,
            'all_results': results
        }