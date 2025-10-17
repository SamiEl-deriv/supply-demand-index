"""
Strategy Backtest System for Supply-Demand Index
Random period selection with multiple strategy backtesting

This system:
1. Selects random 5-day periods from MT5 Vol 75 data
2. Calculates exposure and generates supply-demand index simulations
3. Defines basic trading strategies (mostly ineffective by design)
4. Backtests each strategy against each simulation
5. Evaluates performance with comprehensive risk metrics

Author: Cline
Date: 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Union
import random
from new_supply_demand_index_engine import NewSupplyDemandIndexEngine
import warnings

warnings.filterwarnings("ignore")

# Set plot style
plt.style.use("ggplot")
plt.rcParams.update({'font.size': 11})


class BasicTradingStrategies:
    """
    Collection of basic trading strategies that are designed to be mostly ineffective
    These represent common retail trading approaches that typically fail
    """
    
    @staticmethod
    def simple_momentum(prices: np.ndarray, lookback: int = 5) -> str:
        """
        Simple momentum strategy - buy if price is rising, sell if falling
        This strategy typically fails due to noise and whipsaws
        """
        if len(prices) < lookback + 1:
            return "hold"
        
        recent_return = (prices[-1] - prices[-lookback-1]) / prices[-lookback-1]
        
        if recent_return > 0.001:  # 0.1% threshold (common)
            return "buy"
        elif recent_return < -0.001:
            return "sell"
        else:
            return "hold"
    
    @staticmethod
    def mean_reversion(prices: np.ndarray, lookback: int = 10) -> str:
        """
        Mean reversion strategy - buy when below average, sell when above
        Often fails due to trending markets and poor timing
        """
        if len(prices) < lookback:
            return "hold"
        
        mean_price = np.mean(prices[-lookback:])
        current_price = prices[-1]
        
        deviation = (current_price - mean_price) / mean_price
        
        if deviation < -0.002:  # 0.2% below mean (common threshold)
            return "buy"
        elif deviation > 0.002:  # 0.2% above mean (common threshold)
            return "sell"
        else:
            return "hold"
    
    @staticmethod
    def rsi_strategy(prices: np.ndarray, period: int = 14) -> str:
        """
        RSI-based strategy - classic overbought/oversold approach
        Often fails due to trending markets and false signals
        """
        if len(prices) < period + 1:
            return "hold"
        
        # Calculate RSI
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return "hold"
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        if rsi < 30:  # Oversold
            return "buy"
        elif rsi > 70:  # Overbought
            return "sell"
        else:
            return "hold"
    
    @staticmethod
    def breakout_strategy(prices: np.ndarray, lookback: int = 20) -> str:
        """
        Breakout strategy - buy on highs, sell on lows
        Often fails due to false breakouts and late entries
        """
        if len(prices) < lookback:
            return "hold"
        
        recent_prices = prices[-lookback:]
        current_price = prices[-1]
        
        high = np.max(recent_prices[:-1])  # Exclude current price
        low = np.min(recent_prices[:-1])
        
        if current_price > high * 1.001:  # 0.1% above high (common)
            return "buy"
        elif current_price < low * 0.999:  # 0.1% below low (common)
            return "sell"
        else:
            return "hold"
    
    @staticmethod
    def contrarian_strategy(prices: np.ndarray, lookback: int = 3) -> str:
        """
        Contrarian strategy - do opposite of recent moves
        Often fails due to momentum and trend persistence
        """
        if len(prices) < lookback + 1:
            return "hold"
        
        recent_return = (prices[-1] - prices[-lookback-1]) / prices[-lookback-1]
        
        # Do opposite of recent move
        if recent_return > 0.003:  # If up 0.3%, sell (common threshold)
            return "sell"
        elif recent_return < -0.003:  # If down 0.3%, buy (common threshold)
            return "buy"
        else:
            return "hold"
    
    @staticmethod
    def random_strategy(prices: np.ndarray, seed: Optional[int] = None) -> str:
        """
        Random strategy - completely random decisions
        Expected to have ~50% success rate with negative returns due to costs
        """
        if seed is not None:
            np.random.seed(seed)
        
        choice = np.random.choice(['buy', 'sell', 'hold'], p=[0.3, 0.3, 0.4])
        return choice


class StrategyBacktester:
    """
    Backtesting engine for trading strategies on supply-demand index simulations
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 position_size: float = 10000,
                 transaction_cost: float = 0.001,  # 0.1% transaction cost
                 slippage: float = 0.0005):        # 0.05% slippage
        """
        Initialize the backtester
        
        Parameters:
        -----------
        initial_capital : float
            Starting capital
        position_size : float
            Size of each position
        transaction_cost : float
            Transaction cost as percentage
        slippage : float
            Slippage as percentage
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        self.slippage = slippage
    
    def backtest_strategy(self, 
                         strategy_func: callable,
                         price_path: np.ndarray,
                         strategy_name: str = "Unknown",
                         random_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Backtest a single strategy on a price path
        
        Parameters:
        -----------
        strategy_func : callable
            Strategy function that takes prices and returns signal
        price_path : np.ndarray
            Price path to backtest on
        strategy_name : str
            Name of the strategy
        random_seed : int, optional
            Random seed for reproducible results
            
        Returns:
        --------
        Dict[str, Any]
            Backtest results
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize tracking variables
        capital = self.initial_capital
        position = 0  # Current position size
        entry_price = 0
        trades = []
        equity_curve = [capital]
        positions = [0]
        signals = []
        
        # Run backtest with 1-tick delay between signal and execution
        previous_signal = "hold"
        
        for i in range(1, len(price_path)):
            current_price = price_path[i]
            
            # Get strategy signal
            if strategy_name == "random_strategy":
                signal = strategy_func(price_path[:i+1], seed=random_seed + i if random_seed else None)
            else:
                signal = strategy_func(price_path[:i+1])
            
            signals.append(signal)
            
            # Execute trades based on PREVIOUS signal (1-tick delay)
            if previous_signal == "buy" and position <= 0:
                # Close short position if any
                if position < 0:
                    exit_price = current_price * (1 + self.slippage)  # Slippage on exit
                    pnl = (entry_price - exit_price) * abs(position)
                    transaction_cost = abs(position) * exit_price * self.transaction_cost
                    net_pnl = pnl - transaction_cost
                    capital += net_pnl
                    
                    trades.append({
                        'type': 'close_short',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': abs(position),
                        'pnl': pnl,
                        'transaction_cost': transaction_cost,
                        'net_pnl': net_pnl,
                        'timestamp': i
                    })
                
                # Open long position
                entry_price = current_price * (1 + self.slippage)  # Slippage on entry
                position = self.position_size / entry_price
                transaction_cost = position * entry_price * self.transaction_cost
                capital -= transaction_cost
                
            elif previous_signal == "sell" and position >= 0:
                # Close long position if any
                if position > 0:
                    exit_price = current_price * (1 - self.slippage)  # Slippage on exit
                    pnl = (exit_price - entry_price) * position
                    transaction_cost = position * exit_price * self.transaction_cost
                    net_pnl = pnl - transaction_cost
                    capital += net_pnl
                    
                    trades.append({
                        'type': 'close_long',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': position,
                        'pnl': pnl,
                        'transaction_cost': transaction_cost,
                        'net_pnl': net_pnl,
                        'timestamp': i
                    })
                
                # Open short position
                entry_price = current_price * (1 - self.slippage)  # Slippage on entry
                position = -self.position_size / entry_price
                transaction_cost = abs(position) * entry_price * self.transaction_cost
                capital -= transaction_cost
                
            elif previous_signal == "hold" and position != 0:
                # Close any existing position
                if position > 0:  # Close long
                    exit_price = current_price * (1 - self.slippage)
                    pnl = (exit_price - entry_price) * position
                    transaction_cost = position * exit_price * self.transaction_cost
                    net_pnl = pnl - transaction_cost
                    capital += net_pnl
                    
                    trades.append({
                        'type': 'close_long',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': position,
                        'pnl': pnl,
                        'transaction_cost': transaction_cost,
                        'net_pnl': net_pnl,
                        'timestamp': i
                    })
                    
                elif position < 0:  # Close short
                    exit_price = current_price * (1 + self.slippage)
                    pnl = (entry_price - exit_price) * abs(position)
                    transaction_cost = abs(position) * exit_price * self.transaction_cost
                    net_pnl = pnl - transaction_cost
                    capital += net_pnl
                    
                    trades.append({
                        'type': 'close_short',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position_size': abs(position),
                        'pnl': pnl,
                        'transaction_cost': transaction_cost,
                        'net_pnl': net_pnl,
                        'timestamp': i
                    })
                
                position = 0
            
            # Calculate current equity (including unrealized PnL)
            if position != 0:
                if position > 0:  # Long position
                    unrealized_pnl = (current_price - entry_price) * position
                else:  # Short position
                    unrealized_pnl = (entry_price - current_price) * abs(position)
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital
            
            equity_curve.append(current_equity)
            positions.append(position)
            
            # Update previous signal for next iteration (1-tick delay)
            previous_signal = signal
        
        # Close final position if any
        if position != 0:
            final_price = price_path[-1]
            if position > 0:  # Close long
                exit_price = final_price * (1 - self.slippage)
                pnl = (exit_price - entry_price) * position
                transaction_cost = position * exit_price * self.transaction_cost
                net_pnl = pnl - transaction_cost
                capital += net_pnl
                
                trades.append({
                    'type': 'close_long',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': position,
                    'pnl': pnl,
                    'transaction_cost': transaction_cost,
                    'net_pnl': net_pnl,
                    'timestamp': len(price_path) - 1
                })
                
            else:  # Close short
                exit_price = final_price * (1 + self.slippage)
                pnl = (entry_price - exit_price) * abs(position)
                transaction_cost = abs(position) * exit_price * self.transaction_cost
                net_pnl = pnl - transaction_cost
                capital += net_pnl
                
                trades.append({
                    'type': 'close_short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'position_size': abs(position),
                    'pnl': pnl,
                    'transaction_cost': transaction_cost,
                    'net_pnl': net_pnl,
                    'timestamp': len(price_path) - 1
                })
        
        # Calculate performance metrics
        final_capital = capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        if len(trades) > 0:
            winning_trades = [t for t in trades if t['net_pnl'] > 0]
            losing_trades = [t for t in trades if t['net_pnl'] < 0]
            
            win_rate = len(winning_trades) / len(trades)
            avg_win = np.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['net_pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(sum([t['net_pnl'] for t in winning_trades])) / abs(sum([t['net_pnl'] for t in losing_trades])) if losing_trades else float('inf')
            
            total_transaction_costs = sum([t['transaction_cost'] for t in trades])
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            total_transaction_costs = 0
        
        # Calculate drawdown
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        if len(equity_curve) > 1:
            returns = np.diff(equity_array) / equity_array[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'strategy_name': strategy_name,
            'total_return': total_return,
            'final_capital': final_capital,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_transaction_costs': total_transaction_costs,
            'equity_curve': equity_curve,
            'trades': trades,
            'positions': positions,
            'signals': signals
        }


def load_mt5_vol75_data(file_path: str = "mt5_vol_75_20250101_20250831.csv") -> pd.DataFrame:
    """Load MT5 Vol 75 position data from CSV file"""
    print(f"Loading MT5 Vol 75 data from: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        df['minutes'] = pd.to_datetime(df['minutes'])
        df = df.sort_values('minutes').reset_index(drop=True)
        
        print(f"Loaded {len(df):,} position records")
        print(f"Date range: {df['minutes'].min()} to {df['minutes'].max()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def select_random_period(df: pd.DataFrame, period_days: int = 12, random_seed: int = 42) -> pd.DataFrame:
    """
    Select a random period from the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    period_days : int
        Number of days to select (default: 12)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Period data
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Get date range
    start_date = df['minutes'].min().date()
    end_date = df['minutes'].max().date()
    
    # Calculate total days available
    total_days = (end_date - start_date).days
    print(f"Total days available: {total_days}")
    
    if total_days < period_days:
        raise ValueError(f"Dataset must contain at least {period_days} days of data")
    
    # Select random start date ensuring we have enough days
    max_start_day = total_days - period_days
    random_day = random.randint(0, max_start_day)
    period_start_date = start_date + timedelta(days=random_day)
    period_end_date = period_start_date + timedelta(days=period_days)
    
    print(f"Selected {period_days}-day period: {period_start_date} to {period_end_date}")
    
    # Extract data for this period
    period_start = pd.Timestamp(period_start_date, tz='UTC')
    period_end = pd.Timestamp(period_end_date, tz='UTC')
    
    period_data = df[
        (df['minutes'] >= period_start) & 
        (df['minutes'] < period_end)
    ].copy()
    
    print(f"Period contains {len(period_data):,} records")
    
    return period_data


def calculate_net_exposure_from_positions(df: pd.DataFrame) -> pd.Series:
    """Calculate net exposure from LONG/SHORT position data"""
    print("Calculating net exposure from position data...")
    
    # Group by minute and position type
    grouped = df.groupby(['minutes', 'position'])['volume_usd'].sum().unstack(fill_value=0)
    
    # Calculate net exposure (LONG - SHORT)
    if 'LONG' in grouped.columns and 'SHORT' in grouped.columns:
        net_exposure = grouped['LONG'] - grouped['SHORT']
    elif 'LONG' in grouped.columns:
        net_exposure = grouped['LONG']
    elif 'SHORT' in grouped.columns:
        net_exposure = -grouped['SHORT']
    else:
        raise ValueError("No LONG or SHORT positions found in data")
    
    print(f"Generated {len(net_exposure)} minute-level exposure points")
    print(f"Exposure range: {net_exposure.min():,.0f} to {net_exposure.max():,.0f}")
    
    return net_exposure


def generate_multiple_simulations_per_strategy(engine: NewSupplyDemandIndexEngine,
                                              exposure_series: pd.Series,
                                              strategies: Dict[str, callable],
                                              num_simulations_per_strategy: int = 10,
                                              random_seed: int = 42) -> Dict[str, List[np.ndarray]]:
    """
    Generate multiple supply-demand simulations for each strategy
    
    Parameters:
    -----------
    engine : NewSupplyDemandIndexEngine
        Engine instance
    exposure_series : pd.Series
        Exposure time series
    strategies : Dict[str, callable]
        Dictionary of strategy functions
    num_simulations_per_strategy : int
        Number of simulations to generate per strategy
    random_seed : int
        Base random seed
        
    Returns:
    --------
    Dict[str, List[np.ndarray]]
        Dictionary mapping strategy names to lists of their dedicated price paths (minute-level)
    """
    print(f"Generating {num_simulations_per_strategy} simulations for each of {len(strategies)} strategies...")
    
    simulations = {}
    
    for i, strategy_name in enumerate(strategies.keys()):
        print(f"  Generating {num_simulations_per_strategy} simulations for {strategy_name} ({i+1}/{len(strategies)})")
        
        strategy_simulations = []
        
        for sim_idx in range(num_simulations_per_strategy):
            print(f"    Simulation {sim_idx+1}/{num_simulations_per_strategy}")
            
            # Generate dynamic path using the engine (second-level)
            # Use a unique seed for each strategy and simulation combination
            unique_seed = random_seed + i * 10000 + sim_idx * 1000 + np.random.randint(0, 1000)
            path_seconds, drift_path, smoothed_drift_path, probability_path = engine.generate_dynamic_exposure_path(
                exposure_series=exposure_series,
                random_seed=unique_seed,
                ma_window=12,
                seconds_per_minute=60
            )
            
            # Sample to minute-level (take every 60th point)
            path_minutes = path_seconds[::60]  # Sample every 60 seconds = 1 minute
            
            # Ensure we have the right length
            expected_length = len(exposure_series) + 1  # +1 for initial price
            if len(path_minutes) > expected_length:
                path_minutes = path_minutes[:expected_length]
            elif len(path_minutes) < expected_length:
                # Extend if needed
                path_minutes = np.concatenate([path_minutes, [path_minutes[-1]] * (expected_length - len(path_minutes))])
            
            strategy_simulations.append(path_minutes)
        
        simulations[strategy_name] = strategy_simulations
        
        # Print aggregate stats for this strategy's simulations
        all_returns = []
        price_ranges = []
        for sim in strategy_simulations:
            returns = np.diff(sim) / sim[:-1]
            all_returns.extend(returns)
            price_ranges.append((sim.min(), sim.max()))
        
        min_price = min([r[0] for r in price_ranges])
        max_price = max([r[1] for r in price_ranges])
        
        print(f"    Aggregate stats:")
        print(f"      Price range: {min_price:.2f} - {max_price:.2f}")
        print(f"      Return std: {np.std(all_returns):.4f} ({np.std(all_returns)*100:.2f}%)")
        print(f"      Max 1-min return: {np.max(np.abs(all_returns)):.4f} ({np.max(np.abs(all_returns))*100:.2f}%)")
    
    total_simulations = len(strategies) * num_simulations_per_strategy
    print(f"Generated {total_simulations} total simulations ({num_simulations_per_strategy} per strategy)")
    return simulations


def run_strategy_backtests(simulations: List[np.ndarray],
                          strategies: Dict[str, callable],
                          backtester: StrategyBacktester,
                          random_seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run backtests for all strategies on all simulations
    
    Parameters:
    -----------
    simulations : List[np.ndarray]
        List of price path simulations
    strategies : Dict[str, callable]
        Dictionary of strategy functions
    backtester : StrategyBacktester
        Backtester instance
    random_seed : int
        Base random seed
        
    Returns:
    --------
    Dict[str, List[Dict[str, Any]]]
        Results for each strategy on each simulation
    """
    print(f"Running backtests for {len(strategies)} strategies on {len(simulations)} simulations...")
    
    results = {}
    
    for strategy_name, strategy_func in strategies.items():
        print(f"\n  Testing strategy: {strategy_name}")
        strategy_results = []
        
        for i, simulation in enumerate(simulations):
            print(f"    Simulation {i+1}/{len(simulations)}")
            
            # Run backtest
            result = backtester.backtest_strategy(
                strategy_func=strategy_func,
                price_path=simulation,
                strategy_name=strategy_name,
                random_seed=random_seed + i
            )
            
            result['simulation_id'] = i
            strategy_results.append(result)
        
        results[strategy_name] = strategy_results
        
        # Print summary for this strategy
        avg_return = np.mean([r['total_return'] for r in strategy_results])
        win_rate = np.mean([r['win_rate'] for r in strategy_results])
        avg_trades = np.mean([r['total_trades'] for r in strategy_results])
        
        print(f"    Average return: {avg_return:.2%}")
        print(f"    Average win rate: {win_rate:.1%}")
        print(f"    Average trades: {avg_trades:.1f}")
    
    return results


def analyze_backtest_results(results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Analyze and summarize backtest results
    
    Parameters:
    -----------
    results : Dict[str, List[Dict[str, Any]]]
        Backtest results for all strategies
        
    Returns:
    --------
    Dict[str, Any]
        Summary analysis
    """
    print("\nAnalyzing backtest results...")
    
    analysis = {}
    
    for strategy_name, strategy_results in results.items():
        returns = [r['total_return'] for r in strategy_results]
        win_rates = [r['win_rate'] for r in strategy_results]
        sharpe_ratios = [r['sharpe_ratio'] for r in strategy_results]
        max_drawdowns = [r['max_drawdown'] for r in strategy_results]
        
        analysis[strategy_name] = {
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            'avg_win_rate': np.mean(win_rates),
            'avg_sharpe': np.mean(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'profitable_simulations': len([r for r in returns if r > 0]),
            'total_simulations': len(returns),
            'success_rate': len([r for r in returns if r > 0]) / len(returns)
        }
    
    return analysis


def create_backtest_visualizations(results: Dict[str, List[Dict[str, Any]]],
                                  analysis: Dict[str, Any],
                                  period_name: str,
                                  plots_dir: str = "plots/backtest") -> None:
    """
    Create comprehensive backtest visualizations
    
    Parameters:
    -----------
    results : Dict[str, List[Dict[str, Any]]]
        Backtest results
    analysis : Dict[str, Any]
        Analysis summary
    period_name : str
        Period identifier
    plots_dir : str
        Directory to save plots
    """
    # Create plots directory
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # 1. Strategy Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Strategy Backtest Results - {period_name}', fontsize=16, fontweight='bold')
    
    strategy_names = list(analysis.keys())
    
    # Average Returns
    ax = axes[0, 0]
    avg_returns = [analysis[s]['avg_return'] for s in strategy_names]
    colors = plt.cm.RdYlGn([(r + 0.1) / 0.2 for r in avg_returns])  # Color based on performance
    bars = ax.bar(strategy_names, avg_returns, color=colors)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title('Average Returns by Strategy')
    ax.set_ylabel('Return (%)')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_returns):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.001),
                f'{value:.1%}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # Win Rates
    ax = axes[0, 1]
    win_rates = [analysis[s]['avg_win_rate'] for s in strategy_names]
    ax.bar(strategy_names, win_rates, color='lightblue', alpha=0.7)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% (Random)')
    ax.set_title('Average Win Rates by Strategy')
    ax.set_ylabel('Win Rate')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    
    # Sharpe Ratios
    ax = axes[1, 0]
    sharpe_ratios = [analysis[s]['avg_sharpe'] for s in strategy_names]
    ax.bar(strategy_names, sharpe_ratios, color='orange', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_title('Average Sharpe Ratios by Strategy')
    ax.set_ylabel('Sharpe Ratio')
    ax.tick_params(axis='x', rotation=45)
    
    # Max Drawdowns
    ax = axes[1, 1]
    max_drawdowns = [analysis[s]['avg_max_drawdown'] for s in strategy_names]
    ax.bar(strategy_names, max_drawdowns, color='red', alpha=0.7)
    ax.set_title('Average Max Drawdowns by Strategy')
    ax.set_ylabel('Max Drawdown')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{period_name}_strategy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Return Distributions
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Return Distributions - {period_name}', fontsize=16, fontweight='bold')
    
    for i, strategy_name in enumerate(strategy_names):
        if i >= 6:  # Limit to 6 strategies
            break
        
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        returns = [r['total_return'] for r in results[strategy_name]]
        ax.hist(returns, bins=10, alpha=0.7, color=colors[i])
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=np.mean(returns), color='blue', linestyle='-', alpha=0.7, label=f'Mean: {np.mean(returns):.1%}')
        ax.set_title(f'{strategy_name}')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(strategy_names), 6):
        row = i // 3
        col = i % 3
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/{period_name}_return_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Backtest visualizations saved to {plots_dir}/")


def main():
    """
    Main function to run the strategy backtest system
    """
    print("=" * 80)
    print("STRATEGY BACKTEST SYSTEM")
    print("Random Period Selection with Multiple Strategy Testing")
    print("=" * 80)
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    random_seed = 42
    num_simulations = 15
    
    # Step 1: Load MT5 Vol 75 data
    print("STEP 1: Loading MT5 Vol 75 data...")
    print("-" * 40)
    try:
        df = load_mt5_vol75_data()
        print(f"✓ Successfully loaded {len(df):,} records")
        print()
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Step 2: Select random period
    print("STEP 2: Selecting random period...")
    print("-" * 40)
    try:
        period_data = select_random_period(df, period_days=12, random_seed=random_seed)
        period_name = f"period_{period_data['minutes'].min().strftime('%Y%m%d')}_{period_data['minutes'].max().strftime('%Y%m%d')}"
        print(f"✓ Selected period: {period_name}")
        print()
    except Exception as e:
        print(f"✗ Error selecting period: {e}")
        return
    
    # Step 3: Calculate net exposure
    print("STEP 3: Calculating net exposure...")
    print("-" * 40)
    try:
        exposure_series = calculate_net_exposure_from_positions(period_data)
        print(f"✓ Calculated exposure for {len(exposure_series)} time points")
        print()
    except Exception as e:
        print(f"✗ Error calculating exposure: {e}")
        return
    
    # Step 4: Initialize supply-demand engine
    print("STEP 4: Initializing supply-demand engine...")
    print("-" * 40)
    try:
        engine = NewSupplyDemandIndexEngine(
            sigma=0.30,
            scale=150_000,
            k=0.40,
            smoothness_factor=2.0,
            noise_injection_level=0.01
        )
        print("✓ Engine initialized with parameters:")
        print(f"  • σ (volatility): {engine.sigma:.1%}")
        print(f"  • Scale: {engine.scale:,}")
        print(f"  • k (range): ±{engine.k:.1%}")
        print(f"  • Smoothness: {engine.smoothness_factor}")
        print(f"  • Noise injection: {engine.noise_injection_level:.1%}")
        print()
    except Exception as e:
        print(f"✗ Error initializing engine: {e}")
        return
    
    # Step 5: Define trading strategies
    print("STEP 5: Defining trading strategies...")
    print("-" * 40)
    strategies = {
        'simple_momentum': BasicTradingStrategies.simple_momentum,
        'mean_reversion': BasicTradingStrategies.mean_reversion,
        'rsi_strategy': BasicTradingStrategies.rsi_strategy,
        'breakout_strategy': BasicTradingStrategies.breakout_strategy,
        'contrarian_strategy': BasicTradingStrategies.contrarian_strategy,
        'random_strategy': BasicTradingStrategies.random_strategy
    }
    print(f"✓ Defined {len(strategies)} trading strategies:")
    for strategy_name in strategies.keys():
        print(f"  • {strategy_name}")
    print()
    
    # Step 6: Generate multiple simulations per strategy
    print("STEP 6: Generating multiple simulations per strategy...")
    print("-" * 40)
    try:
        simulations = generate_multiple_simulations_per_strategy(
            engine=engine,
            exposure_series=exposure_series,
            strategies=strategies,
            num_simulations_per_strategy=10,
            random_seed=random_seed
        )
        print(f"✓ Generated simulations for all strategies")
        print()
    except Exception as e:
        print(f"✗ Error generating simulations: {e}")
        return
    
    # Step 7: Initialize backtester
    print("STEP 7: Initializing backtester...")
    print("-" * 40)
    try:
        backtester = StrategyBacktester(
            initial_capital=100_000,
            position_size=10_000,
            transaction_cost=0.0005,  # 0.05%
            slippage=0.0             # No slippage
        )
        print("✓ Backtester initialized:")
        print(f"  • Initial capital: ${backtester.initial_capital:,}")
        print(f"  • Position size: ${backtester.position_size:,}")
        print(f"  • Transaction cost: {backtester.transaction_cost:.2%}")
        print(f"  • Slippage: {backtester.slippage:.2%}")
        print()
    except Exception as e:
        print(f"✗ Error initializing backtester: {e}")
        return
    
    # Step 8: Run strategy backtests on multiple simulations
    print("STEP 8: Running strategy backtests on multiple simulations...")
    print("-" * 40)
    try:
        results = {}
        
        for strategy_name, strategy_func in strategies.items():
            print(f"\n  Testing strategy: {strategy_name} on 10 simulations")
            
            strategy_results = []
            total_pnl_all_trades = 0
            total_trades_count = 0
            
            # Get all simulations for this strategy
            strategy_simulations = simulations[strategy_name]
            
            for sim_idx, price_path in enumerate(strategy_simulations):
                print(f"    Simulation {sim_idx+1}/10")
                
                # Run backtest
                result = backtester.backtest_strategy(
                    strategy_func=strategy_func,
                    price_path=price_path,
                    strategy_name=strategy_name,
                    random_seed=random_seed + sim_idx
                )
                
                result['simulation_id'] = sim_idx
                strategy_results.append(result)
                
                # Calculate sum of all trade PnLs for this simulation
                if result['trades']:
                    sim_total_pnl = sum([trade['net_pnl'] for trade in result['trades']])
                    total_pnl_all_trades += sim_total_pnl
                    total_trades_count += len(result['trades'])
                    print(f"      Trades: {len(result['trades'])}, Total PnL: ${sim_total_pnl:,.2f}")
                else:
                    print(f"      Trades: 0, Total PnL: $0.00")
            
            results[strategy_name] = strategy_results
            
            # Print summary for this strategy across all simulations
            avg_return = np.mean([r['total_return'] for r in strategy_results])
            avg_win_rate = np.mean([r['win_rate'] for r in strategy_results])
            avg_trades = np.mean([r['total_trades'] for r in strategy_results])
            
            print(f"  STRATEGY SUMMARY:")
            print(f"    Average return: {avg_return:.2%}")
            print(f"    Average win rate: {avg_win_rate:.1%}")
            print(f"    Average trades per simulation: {avg_trades:.1f}")
            print(f"    Total trades across all simulations: {total_trades_count}")
            print(f"    SUM OF ALL TRADE PnLs: ${total_pnl_all_trades:,.2f}")
        
        print(f"\n✓ Completed backtests for all strategies")
        print()
    except Exception as e:
        print(f"✗ Error running backtests: {e}")
        return
    
    # Step 9: Analyze results
    print("STEP 9: Analyzing backtest results...")
    print("-" * 40)
    try:
        analysis = analyze_backtest_results(results)
        print("✓ Analysis completed")
        print()
    except Exception as e:
        print(f"✗ Error analyzing results: {e}")
        return
    
    # Step 10: Create visualizations
    print("STEP 10: Creating visualizations...")
    print("-" * 40)
    try:
        create_backtest_visualizations(
            results=results,
            analysis=analysis,
            period_name=period_name
        )
        print("✓ Visualizations created")
        print()
    except Exception as e:
        print(f"✗ Error creating visualizations: {e}")
        return
    
    # Step 11: Print final summary
    print("STEP 11: Final Summary")
    print("-" * 40)
    print("Strategy Performance Summary:")
    print()
    
    for strategy_name, stats in analysis.items():
        print(f"{strategy_name}:")
        print(f"  • Average return: {stats['avg_return']:.2%}")
        print(f"  • Win rate: {stats['avg_win_rate']:.1%}")
        print(f"  • Sharpe ratio: {stats['avg_sharpe']:.2f}")
        print(f"  • Max drawdown: {stats['avg_max_drawdown']:.2%}")
        print(f"  • Success rate: {stats['success_rate']:.1%}")
        print()
    
    # Identify best and worst strategies
    best_strategy = max(analysis.keys(), key=lambda x: analysis[x]['avg_return'])
    worst_strategy = min(analysis.keys(), key=lambda x: analysis[x]['avg_return'])
    
    print("Key Findings:")
    print(f"  • Best performing strategy: {best_strategy} ({analysis[best_strategy]['avg_return']:.2%})")
    print(f"  • Worst performing strategy: {worst_strategy} ({analysis[worst_strategy]['avg_return']:.2%})")
    print(f"  • Period analyzed: {period_name}")
    print(f"  • Total simulations: {num_simulations}")
    print(f"  • Total strategies tested: {len(strategies)}")
    
    print()
    print("=" * 80)
    print("STRATEGY BACKTEST ANALYSIS COMPLETED!")
    print("=" * 80)
    print(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Check 'plots/backtest/' directory for visualizations.")


if __name__ == "__main__":
    main()
