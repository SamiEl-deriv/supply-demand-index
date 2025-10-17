"""
New Vol 75 Supply-Demand Index Analysis for MT5 Data

This script analyzes MT5 Vol 75 position data using the new supply-demand index engine
with random 7-day period selection and enhanced mathematical models.

Key Features:
- Processes MT5 minute-level LONG/SHORT position data
- Random selection of multiple 7-day periods for analysis
- Enhanced supply-demand index engine with noise injection
- Comprehensive visualization and validation
- Exploitation risk assessment

Author: Cline
Date: 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Union
from new_supply_demand_index_engine import NewSupplyDemandIndexEngine
import random

# Set plot style
plt.style.use("ggplot")
plt.rcParams.update({'font.size': 11})


def load_mt5_vol75_data(
    file_path: str = "mt5_vol_75_20250101_20250831.csv"
) -> pd.DataFrame:
    """
    Load MT5 Vol 75 position data from CSV file

    Parameters:
    -----------
    file_path : str
        Path to the MT5 Vol 75 data file

    Returns:
    --------
    pd.DataFrame
        DataFrame with minute-level position data
    """
    print(f"Loading MT5 Vol 75 data from: {file_path}")
    
    try:
        # Load the data
        df = pd.read_csv(file_path)
        
        # Convert timestamp
        df['minutes'] = pd.to_datetime(df['minutes'])
        
        # Sort by timestamp
        df = df.sort_values('minutes').reset_index(drop=True)
        
        print(f"Loaded {len(df):,} position records")
        print(f"Date range: {df['minutes'].min()} to {df['minutes'].max()}")
        print(f"Unique positions: {df['position'].unique()}")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def select_mixed_periods(
    df: pd.DataFrame, 
    num_short_periods: int = 4,
    num_long_periods: int = 1,
    short_period_days: int = 7,
    long_period_days: int = 60,
    random_seed: int = 42
) -> List[Tuple[pd.DataFrame, str]]:
    """
    Select mixed periods: short (7-day) and long (2-month) periods from the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    num_short_periods : int
        Number of short periods to select
    num_long_periods : int
        Number of long periods to select
    short_period_days : int
        Days for short periods
    long_period_days : int
        Days for long periods (60 days ≈ 2 months)
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    List[Tuple[pd.DataFrame, str]]
        List of tuples (DataFrame, period_type) where period_type is "short" or "long"
    """
    print(f"Selecting {num_short_periods} short periods ({short_period_days} days) + {num_long_periods} long periods ({long_period_days} days)...")
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Get date range
    start_date = df['minutes'].min().date()
    end_date = df['minutes'].max().date()
    
    # Calculate total days available
    total_days = (end_date - start_date).days
    print(f"Total days available: {total_days}")
    
    if total_days < max(short_period_days, long_period_days):
        raise ValueError(f"Dataset must contain at least {max(short_period_days, long_period_days)} days of data")
    
    selected_periods = []
    selected_dates = []
    
    # First, select short periods (7 days each)
    print(f"\nSelecting {num_short_periods} short periods ({short_period_days} days each):")
    max_start_day_short = total_days - short_period_days
    
    for i in range(num_short_periods):
        attempts = 0
        while attempts < 100:
            random_day = random.randint(0, max_start_day_short)
            period_start_date = start_date + timedelta(days=random_day)
            period_end_date = period_start_date + timedelta(days=short_period_days)
            
            # Check for overlaps
            overlap = False
            for existing_start, existing_end in selected_dates:
                if (period_start_date < existing_end and period_end_date > existing_start):
                    overlap = True
                    break
            
            if not overlap:
                break
            attempts += 1
        
        if attempts >= 100:
            print(f"Warning: Could not find non-overlapping short period for selection {i+1}")
            random_day = random.randint(0, max_start_day_short)
            period_start_date = start_date + timedelta(days=random_day)
            period_end_date = period_start_date + timedelta(days=short_period_days)
        
        selected_dates.append((period_start_date, period_end_date))
        
        # Extract data
        period_start = pd.Timestamp(period_start_date, tz='UTC')
        period_end = pd.Timestamp(period_end_date, tz='UTC')
        
        period_data = df[
            (df['minutes'] >= period_start) & 
            (df['minutes'] < period_end)
        ].copy()
        
        if len(period_data) > 0:
            selected_periods.append((period_data, "short"))
            print(f"  Short Period {i+1}: {period_start_date} to {period_end_date} ({len(period_data):,} records)")
    
    # Then, select long periods (60 days each)
    print(f"\nSelecting {num_long_periods} long periods ({long_period_days} days each):")
    max_start_day_long = total_days - long_period_days
    
    for i in range(num_long_periods):
        attempts = 0
        while attempts < 100:
            random_day = random.randint(0, max_start_day_long)
            period_start_date = start_date + timedelta(days=random_day)
            period_end_date = period_start_date + timedelta(days=long_period_days)
            
            # Check for overlaps with existing periods
            overlap = False
            for existing_start, existing_end in selected_dates:
                if (period_start_date < existing_end and period_end_date > existing_start):
                    overlap = True
                    break
            
            if not overlap:
                break
            attempts += 1
        
        if attempts >= 100:
            print(f"Warning: Could not find non-overlapping long period for selection {i+1}")
            random_day = random.randint(0, max_start_day_long)
            period_start_date = start_date + timedelta(days=random_day)
            period_end_date = period_start_date + timedelta(days=long_period_days)
        
        selected_dates.append((period_start_date, period_end_date))
        
        # Extract data
        period_start = pd.Timestamp(period_start_date, tz='UTC')
        period_end = pd.Timestamp(period_end_date, tz='UTC')
        
        period_data = df[
            (df['minutes'] >= period_start) & 
            (df['minutes'] < period_end)
        ].copy()
        
        if len(period_data) > 0:
            selected_periods.append((period_data, "long"))
            print(f"  Long Period {i+1}: {period_start_date} to {period_end_date} ({len(period_data):,} records)")
    
    print(f"\nSuccessfully selected {len(selected_periods)} periods total")
    return selected_periods


def calculate_net_exposure_from_positions(
    df: pd.DataFrame
) -> pd.Series:
    """
    Calculate net exposure from LONG/SHORT position data
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns: minutes, position, volume_usd
        
    Returns:
    --------
    pd.Series
        Series of net exposure values indexed by minute
    """
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


def run_analysis_on_period(
    period_data: pd.DataFrame,
    period_name: str,
    sample_size: int = 50,
    random_seed: int = 42
) -> Tuple[NewSupplyDemandIndexEngine, Dict[str, Any], pd.Series]:
    """
    Run supply-demand index analysis on a single 7-day period
    
    Parameters:
    -----------
    period_data : pd.DataFrame
        7-day period data
    period_name : str
        Name/identifier for this period
    sample_size : int
        Number of exposure points to sample for detailed analysis
    random_seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple[NewSupplyDemandIndexEngine, Dict[str, Any], pd.Series]
        Engine instance, results dictionary, and exposure series
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING PERIOD: {period_name}")
    print(f"{'='*60}")
    
    # Calculate net exposure
    exposure_series = calculate_net_exposure_from_positions(period_data)
    
    # Initialize the engine
    engine = NewSupplyDemandIndexEngine(
        sigma=0.30,  # 30% volatility
        scale=150_000,  # Exposure sensitivity
        k=0.40,  # Max probability deviation (±40%)
        T=1.0 / (365 * 24),  # 1 hour time horizon
        S_0=10_000,  # Starting price
        smoothness_factor=2.0,  # Smoothness control
        noise_injection_level=0.01  # 1% noise injection
    )
    
    print("Initialized New Supply-Demand Index Engine:")
    print(f"  σ (volatility): {engine.sigma:.1%}")
    print(f"  Scale: {engine.scale:,}")
    print(f"  k (range): ±{engine.k:.1%}")
    print(f"  Smoothness factor: {engine.smoothness_factor}")
    print(f"  Noise injection: {engine.noise_injection_level:.1%}")
    
    # Sample exposure data for detailed analysis
    np.random.seed(random_seed)
    if len(exposure_series) > sample_size:
        sampled_indices = np.random.choice(
            len(exposure_series), size=sample_size, replace=False
        )
        sampled_exposure = exposure_series.iloc[sampled_indices].values
    else:
        sampled_exposure = exposure_series.values
    
    print(f"Processing {len(sampled_exposure)} sampled exposure points...")
    
    # Process the exposure data
    results = engine.process_exposure_data(
        exposure_data=sampled_exposure,
        duration_in_minutes=60,  # 1 hour simulation per exposure
        num_paths_per_exposure=50,  # 50 Monte Carlo paths
        random_seed=random_seed
    )
    
    return engine, results, exposure_series


def create_period_visualizations(
    engine: NewSupplyDemandIndexEngine,
    results: Dict[str, Any],
    exposure_series: pd.Series,
    period_name: str,
    plots_dir: str = "plots/new"
) -> None:
    """
    Create visualizations for a single period analysis
    
    Parameters:
    -----------
    engine : NewSupplyDemandIndexEngine
        Engine instance
    results : Dict[str, Any]
        Analysis results
    exposure_series : pd.Series
        Full exposure time series
    period_name : str
        Period identifier
    plots_dir : str
        Directory to save plots
    """
    # Create plots directory
    period_plots_dir = f"{plots_dir}/{period_name}"
    if not os.path.exists(period_plots_dir):
        os.makedirs(period_plots_dir)
        print(f"Created directory: {period_plots_dir}")
    
    # 1. Exposure vs Probability Mapping
    plt.figure(figsize=(12, 8))
    exposures = results["exposures"]
    probabilities = results["probabilities"]
    
    # Create theoretical curve
    exposure_range = np.linspace(min(exposures) * 1.2, max(exposures) * 1.2, 1000)
    theoretical_probs = [engine.exposure_to_probability(exp) for exp in exposure_range]
    
    plt.plot(exposure_range, theoretical_probs, "-", color="blue", linewidth=2.5, label="Theoretical Mapping")
    plt.scatter(exposures, probabilities, color="red", s=50, alpha=0.8, label="Data Points")
    plt.axhline(y=0.5, color="black", linestyle="--", alpha=0.6, label="Neutral (0.5)")
    plt.axvline(x=0, color="black", linestyle=":", alpha=0.6)
    
    plt.title(f"Exposure-to-Probability Mapping - {period_name}", fontsize=16)
    plt.xlabel("Net Exposure ($)", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{period_plots_dir}/1_exposure_probability_mapping.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Time Series of Exposure
    plt.figure(figsize=(16, 6))
    plt.plot(exposure_series.index, exposure_series.values, color="darkgreen", linewidth=1.0)
    plt.title(f"Net Exposure Time Series - {period_name}", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Net Exposure ($)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{period_plots_dir}/2_exposure_timeseries.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Sample Price Paths (first 3 scenarios)
    n_scenarios = min(3, len(results["exposures"]))
    for i in range(n_scenarios):
        plt.figure(figsize=(10, 6))
        
        exposure = results["exposures"][i]
        paths = results["price_paths"][i]
        mean_path = results["mean_paths"][i]
        
        # Plot sample paths
        for j, path in enumerate(paths[:20]):  # Show first 20 paths
            plt.plot(path, color="lightblue", alpha=0.3, linewidth=0.5)
        
        plt.plot(mean_path, color="red", linewidth=2, label="Mean Path")
        plt.axhline(y=engine.S_0, color="black", linestyle="--", alpha=0.5, label="Initial Price")
        
        plt.title(f"Price Paths for Exposure: ${exposure:,.0f} - {period_name}", fontsize=14)
        plt.xlabel("Time Steps (minutes)", fontsize=12)
        plt.ylabel("Price", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{period_plots_dir}/3_{i+1}_price_paths_exp_{int(exposure/1000)}k.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    # 4. Dynamic Index Path with Multiple Simulations
    print(f"  Generating dynamic index path for {period_name}...")
    
    # Run multiple simulations like in adjusted version
    num_simulations = 30
    all_paths = []
    
    # Use period name hash to ensure different seeds for different periods
    base_seed = hash(period_name) % 10000
    
    for i in range(num_simulations):
        path, drift, smoothed_drift, prob = engine.generate_dynamic_exposure_path(
            exposure_series=exposure_series,
            random_seed=base_seed + i,  # Different seed for each simulation
            ma_window=60  # 1 hour moving average
        )
        all_paths.append(path)
    
    # Convert to numpy array and calculate statistics
    all_paths = np.array(all_paths)
    mean_path = np.mean(all_paths, axis=0)
    lower_band = np.percentile(all_paths, 10, axis=0)
    upper_band = np.percentile(all_paths, 90, axis=0)
    
    # Create the plot with 3 subplots: Exposure, Index, and Correlation
    fig, (ax_exp, ax_idx, ax_corr) = plt.subplots(3, 1, figsize=(16, 14), height_ratios=[2, 2, 1])
    
    # Top subplot: Exposure with trend indicators
    exposure_values = exposure_series.values
    time_points = range(len(exposure_values))
    
    # Calculate moving averages for trend detection
    ma_short = pd.Series(exposure_values).rolling(window=20, min_periods=1).mean()
    ma_long = pd.Series(exposure_values).rolling(window=60, min_periods=1).mean()
    
    # Plot exposure and moving averages
    ax_exp.plot(time_points, exposure_values, color="darkgreen", linewidth=1.5, label="Net Exposure", alpha=0.8)
    ax_exp.plot(time_points, ma_short, color="orange", linewidth=1.0, label="MA(20)", alpha=0.7)
    ax_exp.plot(time_points, ma_long, color="purple", linewidth=1.0, label="MA(60)", alpha=0.7)
    ax_exp.axhline(y=0, color="green", linestyle=":", alpha=0.5, label="Zero Exposure")
    
    # Add trend signals
    bullish_signal = ma_short > ma_long
    bearish_signal = ma_short < ma_long
    
    # Fill areas to show trend
    ax_exp.fill_between(time_points, 0, exposure_values, 
                        where=(exposure_values > 0) & bullish_signal, 
                        color='lightgreen', alpha=0.3, label='Bullish Zone')
    ax_exp.fill_between(time_points, 0, exposure_values, 
                        where=(exposure_values < 0) & bearish_signal, 
                        color='lightcoral', alpha=0.3, label='Bearish Zone')
    
    # Configure exposure subplot
    ax_exp.set_ylabel("Net Exposure ($)", color="darkgreen", fontsize=12)
    ax_exp.tick_params(axis='y', labelcolor="darkgreen")
    ax_exp.grid(True, alpha=0.3)
    ax_exp.legend(loc="upper right", fontsize=9)
    ax_exp.set_title("Net Exposure with Trend Indicators", fontsize=12, color="darkgreen")
    
    # Format exposure axis labels
    ax_exp.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Middle subplot: Supply-Demand Index
    # Scale time points to match exposure (convert seconds back to minutes for display)
    index_time_points = np.linspace(0, len(exposure_values)-1, len(mean_path))
    
    # Plot individual paths (sample)
    for i in range(min(20, num_simulations)):
        ax_idx.plot(index_time_points, all_paths[i], alpha=0.15, color="steelblue", linewidth=0.8)
    
    # Plot mean and confidence bands
    ax_idx.plot(index_time_points, mean_path, color="darkblue", linewidth=2.5, label="Mean Index Path")
    ax_idx.fill_between(
        index_time_points,
        lower_band,
        upper_band,
        color="blue",
        alpha=0.2,
        label="80% Confidence Band"
    )
    
    ax_idx.axhline(y=engine.S_0, color="red", linestyle="--", alpha=0.7, label="Starting Price")
    
    # Configure index subplot
    ax_idx.set_ylabel("Supply-Demand Index", color="darkblue", fontsize=12)
    ax_idx.tick_params(axis='y', labelcolor="darkblue")
    ax_idx.grid(True, alpha=0.3)
    ax_idx.legend(loc="upper left", fontsize=9)
    ax_idx.set_title("Dynamic Supply-Demand Index", fontsize=12, color="darkblue")
    
    # Bottom subplot: Correlation analysis
    # Calculate correlation between exposure changes and index changes
    # Resample index changes to match exposure frequency
    exposure_changes = np.diff(exposure_values)
    
    # Downsample index changes to minute-level to match exposure
    index_minute_changes = []
    seconds_per_minute = len(mean_path) // len(exposure_values)
    for i in range(len(exposure_values) - 1):
        start_idx = i * seconds_per_minute
        end_idx = (i + 1) * seconds_per_minute
        if end_idx < len(mean_path):
            minute_change = mean_path[end_idx] - mean_path[start_idx]
            index_minute_changes.append(minute_change)
    
    index_minute_changes = np.array(index_minute_changes)
    
    # Ensure both arrays have the same length
    min_length = min(len(exposure_changes), len(index_minute_changes))
    exposure_changes = exposure_changes[:min_length]
    index_minute_changes = index_minute_changes[:min_length]
    
    # Rolling correlation
    correlation_window = 100
    rolling_corr = []
    for i in range(len(exposure_changes)):
        start_idx = max(0, i - correlation_window + 1)
        end_idx = i + 1
        if end_idx - start_idx >= 10:  # Minimum window size
            if len(exposure_changes[start_idx:end_idx]) > 0 and len(index_minute_changes[start_idx:end_idx]) > 0:
                corr = np.corrcoef(exposure_changes[start_idx:end_idx], 
                                 index_minute_changes[start_idx:end_idx])[0, 1]
                rolling_corr.append(corr if not np.isnan(corr) else 0)
            else:
                rolling_corr.append(0)
        else:
            rolling_corr.append(0)
    
    ax_corr.plot(range(1, len(rolling_corr) + 1), rolling_corr, color="red", linewidth=2, label="Rolling Correlation")
    ax_corr.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax_corr.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="Strong Positive")
    ax_corr.axhline(y=-0.5, color="red", linestyle="--", alpha=0.5, label="Strong Negative")
    ax_corr.fill_between(range(1, len(rolling_corr) + 1), 0, rolling_corr, 
                         where=np.array(rolling_corr) > 0, color='green', alpha=0.3)
    ax_corr.fill_between(range(1, len(rolling_corr) + 1), 0, rolling_corr, 
                         where=np.array(rolling_corr) < 0, color='red', alpha=0.3)
    
    ax_corr.set_xlabel("Time (minutes)", fontsize=12)
    ax_corr.set_ylabel("Correlation", fontsize=12)
    ax_corr.set_ylim(-1, 1)
    ax_corr.grid(True, alpha=0.3)
    ax_corr.legend(fontsize=9)
    ax_corr.set_title("Rolling Correlation: Exposure Changes vs Index Changes", fontsize=12)
    
    plt.suptitle(f"Supply-Demand Analysis - {period_name}\n({num_simulations} simulations)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{period_plots_dir}/4_dynamic_index_path.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"  Created visualizations for {period_name}")


def print_period_summary(
    period_data: pd.DataFrame,
    exposure_series: pd.Series,
    engine: NewSupplyDemandIndexEngine,
    results: Dict[str, Any],
    period_name: str
) -> None:
    """
    Print comprehensive summary for a period analysis
    
    Parameters:
    -----------
    period_data : pd.DataFrame
        Period data
    exposure_series : pd.Series
        Exposure time series
    engine : NewSupplyDemandIndexEngine
        Engine instance
    results : Dict[str, Any]
        Analysis results
    period_name : str
        Period identifier
    """
    print(f"\n{'='*50}")
    print(f"PERIOD SUMMARY: {period_name}")
    print(f"{'='*50}")
    
    # Data statistics
    total_volume = period_data['volume_usd'].sum()
    long_volume = period_data[period_data['position'] == 'LONG']['volume_usd'].sum()
    short_volume = period_data[period_data['position'] == 'SHORT']['volume_usd'].sum()
    
    print(f"Period: {period_data['minutes'].min()} to {period_data['minutes'].max()}")
    print(f"Total Records: {len(period_data):,}")
    print(f"Total Volume: ${total_volume:,.0f}")
    print(f"LONG Volume: ${long_volume:,.0f} ({long_volume/total_volume:.1%})")
    print(f"SHORT Volume: ${short_volume:,.0f} ({short_volume/total_volume:.1%})")
    print(f"Net Exposure Range: ${exposure_series.min():,.0f} to ${exposure_series.max():,.0f}")
    print(f"Exposure Std Dev: ${exposure_series.std():,.0f}")
    
    # Print validation summary
    engine.print_validation_summary(results)


def main():
    """Main function to run the new Vol 75 supply-demand index analysis with mixed periods"""
    print("Starting New Vol 75 Supply-Demand Index Analysis with Mixed Periods...")
    print("=" * 80)
    
    # Load MT5 Vol 75 data
    print("STEP 1: Loading MT5 Vol 75 data...")
    df = load_mt5_vol75_data()
    
    # Select mixed periods: 4 short (7-day) + 1 long (2-month)
    print("\nSTEP 2: Selecting mixed periods...")
    periods_with_types = select_mixed_periods(
        df, 
        num_short_periods=4, 
        num_long_periods=1, 
        short_period_days=7, 
        long_period_days=60, 
        random_seed=42
    )
    
    # Analyze each period
    print("\nSTEP 3: Analyzing each period...")
    all_results = []
    
    for i, (period_data, period_type) in enumerate(periods_with_types, 1):
        period_start = period_data['minutes'].min().strftime('%Y%m%d')
        period_name = f"{period_type}_{i}_{period_start}"
        
        # Adjust sample size based on period type
        if period_type == "long":
            sample_size = 100  # More samples for longer periods
            print(f"\n*** LONG PERIOD ANALYSIS (2 months) ***")
        else:
            sample_size = 50   # Standard samples for short periods
            print(f"\n*** SHORT PERIOD ANALYSIS (7 days) ***")
        
        # Run analysis
        engine, results, exposure_series = run_analysis_on_period(
            period_data=period_data,
            period_name=period_name,
            sample_size=sample_size,
            random_seed=42 + i
        )
        
        # Create visualizations
        create_period_visualizations(
            engine=engine,
            results=results,
            exposure_series=exposure_series,
            period_name=period_name
        )
        
        # Print summary
        print_period_summary(
            period_data=period_data,
            exposure_series=exposure_series,
            engine=engine,
            results=results,
            period_name=period_name
        )
        
        all_results.append({
            'period_name': period_name,
            'period_type': period_type,
            'engine': engine,
            'results': results,
            'exposure_series': exposure_series,
            'period_data': period_data
        })
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    # Separate short and long periods
    short_results = [r for r in all_results if r['period_type'] == 'short']
    long_results = [r for r in all_results if r['period_type'] == 'long']
    
    print(f"Analyzed {len(short_results)} short periods (7 days) + {len(long_results)} long periods (2 months)")
    
    print("\nSHORT PERIODS (7 days):")
    for result in short_results:
        period_name = result['period_name']
        exposure_series = result['exposure_series']
        results = result['results']
        
        avg_validation = np.mean([s['mu_validation_rate'] for s in results['summary_stats']])
        exposure_range = exposure_series.max() - exposure_series.min()
        
        print(f"  {period_name}:")
        print(f"    • Exposure range: ${exposure_range:,.0f}")
        print(f"    • Average validation rate: {avg_validation:.1%}")
        print(f"    • Scenarios analyzed: {len(results['summary_stats'])}")
        print(f"    • Data points: {len(exposure_series):,}")
    
    print("\nLONG PERIODS (2 months):")
    for result in long_results:
        period_name = result['period_name']
        exposure_series = result['exposure_series']
        results = result['results']
        
        avg_validation = np.mean([s['mu_validation_rate'] for s in results['summary_stats']])
        exposure_range = exposure_series.max() - exposure_series.min()
        
        print(f"  {period_name}:")
        print(f"    • Exposure range: ${exposure_range:,.0f}")
        print(f"    • Average validation rate: {avg_validation:.1%}")
        print(f"    • Scenarios analyzed: {len(results['summary_stats'])}")
        print(f"    • Data points: {len(exposure_series):,}")
        print(f"    • Period length: ~{len(exposure_series)/(24*60):.1f} days")
    
    # Compare short vs long period characteristics
    if short_results and long_results:
        print("\nCOMPARISON: Short vs Long Periods")
        print("-" * 40)
        
        # Average validation rates
        short_avg_validation = np.mean([
            np.mean([s['mu_validation_rate'] for s in r['results']['summary_stats']]) 
            for r in short_results
        ])
        long_avg_validation = np.mean([
            np.mean([s['mu_validation_rate'] for s in r['results']['summary_stats']]) 
            for r in long_results
        ])
        
        print(f"Average validation rate:")
        print(f"  Short periods: {short_avg_validation:.1%}")
        print(f"  Long periods:  {long_avg_validation:.1%}")
        
        # Average exposure volatility
        short_avg_volatility = np.mean([r['exposure_series'].std() for r in short_results])
        long_avg_volatility = np.mean([r['exposure_series'].std() for r in long_results])
        
        print(f"Average exposure volatility:")
        print(f"  Short periods: ${short_avg_volatility:,.0f}")
        print(f"  Long periods:  ${long_avg_volatility:,.0f}")
    
    print(f"\nAnalysis complete! Check 'plots/new/' directory for visualizations.")
    print("=" * 80)


if __name__ == "__main__":
    main()
