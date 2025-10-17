"""
Risk-Optimized Vol 75 Supply-Demand Index Analysis

This script analyzes Vol 75 trade data using the risk-optimized supply-demand index engine with:
1. Advanced parameter optimization using Bayesian methods
2. Anti-exploitation mechanisms with controlled randomness
3. Comprehensive backtesting framework for parameter validation
4. Real-time risk monitoring and adaptive parameter adjustment
5. Multi-objective optimization balancing responsiveness, exploitability, and risk

Author: Cline
Date: 2025
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Union
import json
from risk_supply_demand_index_engine import RiskOptimizedSupplyDemandIndexEngine

# Set plot style
plt.style.use("ggplot")
plt.rcParams.update({'font.size': 11})


def load_vol75_trade_data(
    data_dir: str = "/Users/samielmokh/Downloads/data/trade/vol_75",
) -> pd.DataFrame:
    """
    Load Vol 75 trade data from CSV files in the specified directory

    Parameters:
    -----------
    data_dir : str
        Directory containing Vol 75 trade data CSV files

    Returns:
    --------
    pd.DataFrame
        Combined DataFrame of all trade data
    """
    # Get all CSV files
    csv_files = glob.glob(f"{data_dir}/*.csv")
    csv_files.sort()

    print(f"Found {len(csv_files)} CSV files")

    # Load and combine data
    dfs = []
    for i, file in enumerate(csv_files):
        print(f"Loading file {i + 1}/{len(csv_files)}: {os.path.basename(file)}")
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    if not dfs:
        raise ValueError("No data files could be loaded")

    # Combine all data
    combined_df = pd.concat(dfs, ignore_index=True)

    # Convert timestamp with mixed format handling
    combined_df["time_deal"] = pd.to_datetime(combined_df["time_deal"], format="mixed")

    # Sort by timestamp
    combined_df = combined_df.sort_values("time_deal").reset_index(drop=True)

    print(f"Loaded {len(combined_df):,} total trades")
    print(
        f"Date range: {combined_df['time_deal'].min()} to {combined_df['time_deal'].max()}"
    )

    # Create a standardized format for our analysis
    trades_df = pd.DataFrame()
    trades_df["timestamp"] = combined_df["time_deal"]

    # Convert action (0=buy, 1=sell) to direction string
    trades_df["direction"] = combined_df["action"].map({0: "buy", 1: "sell"})

    # Use volume_usd as amount
    trades_df["amount"] = combined_df["volume_usd"]

    return trades_df


def calculate_net_exposure(
    trades_df: pd.DataFrame, window_minutes: int = 5
) -> pd.Series:
    """
    Calculate net exposure in fixed time windows

    Parameters:
    -----------
    trades_df : pd.DataFrame
        DataFrame containing trade data with columns: timestamp, direction, amount
    window_minutes : int
        Size of time window in minutes

    Returns:
    --------
    pd.Series
        Series of net exposure values indexed by window end time
    """
    print(f"Calculating net exposure with {window_minutes}min windows...")

    # Ensure timestamp is datetime
    trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])

    # Create time windows
    start_time = trades_df["timestamp"].min().floor("D")  # Start of day
    end_time = trades_df["timestamp"].max().ceil("D")  # End of day

    # Create window boundaries
    window_size = timedelta(minutes=window_minutes)
    windows = pd.date_range(start=start_time, end=end_time, freq=f"{window_minutes}min")

    # Initialize exposure Series
    exposure = pd.Series(0.0, index=windows[:-1])

    # Group trades by window
    for i in range(len(windows) - 1):
        window_start = windows[i]
        window_end = windows[i + 1]

        # Get trades in this window
        mask = (trades_df["timestamp"] >= window_start) & (
            trades_df["timestamp"] < window_end
        )
        window_trades = trades_df[mask]

        if len(window_trades) > 0:
            # Calculate net exposure (buy - sell)
            buys = window_trades[window_trades["direction"] == "buy"]["amount"].sum()
            sells = window_trades[window_trades["direction"] == "sell"]["amount"].sum()
            net = buys - sells
            exposure[window_start] = net

    # Remove windows with no trades
    exposure = exposure[exposure != 0]

    print(f"Generated {len(exposure)} time windows with trades")
    print(f"Exposure range: {exposure.min():,.0f} to {exposure.max():,.0f}")

    return exposure


def run_risk_optimized_analysis(
    exposure_data: pd.Series, 
    sample_size: int = 50, 
    random_seed: int = 1505,
    optimization_method: str = 'scipy',
    n_optimization_calls: int = 30
) -> Tuple[RiskOptimizedSupplyDemandIndexEngine, Dict[str, Any]]:
    """
    Run risk-optimized supply-demand index analysis on exposure data

    Parameters:
    -----------
    exposure_data : pd.Series
        Series of net exposure values
    sample_size : int
        Number of exposure points to sample for detailed analysis
    random_seed : int
        Random seed for reproducibility
    optimization_method : str
        'bayesian' or 'scipy'
    n_optimization_calls : int
        Number of optimization iterations

    Returns:
    --------
    Tuple[RiskOptimizedSupplyDemandIndexEngine, Dict[str, Any]]
        Engine instance and comprehensive backtest results
    """
    # Initialize the risk-optimized engine with conservative parameters
    engine = RiskOptimizedSupplyDemandIndexEngine(
        sigma=0.35,  # 35% volatility - balanced for crypto assets
        scale=140_000,  # Exposure sensitivity calibrated to Vol 75
        k=0.38,  # Conservative probability deviation (0.12 to 0.88 range)
        T=1.0 / (365 * 24),  # 1 hour time horizon
        S_0=100_000,  # Starting price (100,000)
        smoothness_factor=2.0,  # Controls smoothness of transitions
        # Risk optimization parameters
        sigma_noise_std=0.05,  # 5% daily sigma variation
        scale_adaptation_rate=0.1,  # 10% adaptation rate
        k_oscillation_amplitude=0.05,  # 5% k oscillation
        noise_injection_level=0.015,  # 1.5% noise injection
        pattern_breaking_frequency=0.1,  # 10% pattern breaking
        psychology_bull_factor=0.35,  # Bullish psychology factor
        psychology_bear_factor=0.45,  # Bearish psychology factor (slightly more bearish)
        # Risk constraints
        max_daily_movement=0.05,  # 5% max daily movement
        max_volatility=0.8,  # 80% max volatility
        min_smoothing=0.05,  # 5% minimum smoothing
        max_sensitivity=0.001,  # 0.1% max sensitivity to single trade
    )

    print("Initialized Risk-Optimized Supply-Demand Index Engine with parameters:")
    print(f"  sigma: {engine.sigma_base:.1%}")
    print(f"  scale: {engine.scale_base}")
    print(f"  k: {engine.k_base}")
    print(f"  T: {engine.T * 365 * 24:.1f} hours")
    print(f"  S_0: {engine.S_0}")
    print(f"  smoothness_factor: {engine.smoothness_factor}")
    print(f"  noise_injection_level: {engine.noise_injection_level:.1%}")
    print(f"  max_daily_movement: {engine.max_daily_movement:.1%}")

    # Sample exposure data for detailed analysis
    np.random.seed(random_seed)
    if len(exposure_data) > sample_size:
        sampled_indices = np.random.choice(
            len(exposure_data), size=sample_size, replace=False
        )
        sampled_exposure = exposure_data.iloc[sampled_indices].values
    else:
        sampled_exposure = exposure_data.values

    print(
        f"Preparing historical data from {len(exposure_data)} exposure points..."
    )
    print(f"Sampled {len(sampled_exposure)} exposure points for optimization")
    print(
        f"Exposure range: {min(sampled_exposure):,.0f} to {max(sampled_exposure):,.0f}"
    )

    # Prepare historical data for optimization
    historical_data = {
        'exposures': sampled_exposure.tolist(),
        'timestamps': [exposure_data.index[0] + timedelta(minutes=i*5) for i in range(len(sampled_exposure))]
    }

    # Run comprehensive backtesting with parameter optimization
    print(f"\nRunning comprehensive backtesting with {optimization_method} optimization...")
    print(f"Optimization iterations: {n_optimization_calls}")
    
    backtest_results = engine.run_comprehensive_backtest(
        historical_data=historical_data,
        optimization_method=optimization_method,
        n_optimization_calls=n_optimization_calls
    )

    return engine, backtest_results


def generate_dynamic_index_path_optimized(
    engine: RiskOptimizedSupplyDemandIndexEngine,
    exposure_series: pd.Series,
    num_simulations: int = 50,
    random_seed: int = 1505,
    ma_window: int = 12,  # 1 hour with 5min data (12 points)
    seconds_per_point: int = 1,  # Generate a point every second
) -> None:
    """
    Generate dynamic index paths using the optimized parameters with per-second resolution
    and overlay aggregated exposure data

    Parameters:
    -----------
    engine : RiskOptimizedSupplyDemandIndexEngine
        Initialized and optimized engine instance
    exposure_series : pd.Series
        Time series of exposure values (5-minute windows)
    num_simulations : int
        Number of Monte Carlo simulations to run
    random_seed : int
        Base random seed for reproducibility
    ma_window : int
        Window size for moving average smoothing
    seconds_per_point : int
        Number of seconds between each generated point (1 = every second)
    """
    print(
        f"Generating optimized dynamic index paths with per-second resolution..."
    )
    print(f"Using random seed base: {random_seed} for dynamic path generation")
    print(f"Generating index points every {seconds_per_point} second(s)")

    # Calculate total duration in seconds (5 minutes per exposure point)
    total_duration_seconds = len(exposure_series) * 300  # 300 seconds = 5 minutes
    
    # Create per-second time array
    time_seconds = np.arange(0, total_duration_seconds, seconds_per_point)
    
    # Interpolate exposure data to per-second resolution
    original_time_points = np.arange(0, total_duration_seconds, 300)  # Every 5 minutes
    interpolated_exposure = np.interp(time_seconds, original_time_points, exposure_series.values)
    
    print(
        f"Interpolated exposure data from {len(exposure_series)} points to {len(interpolated_exposure)} points"
    )
    print(f"Total simulation duration: {total_duration_seconds/3600:.1f} hours")

    # Run multiple simulations with optimized parameters
    all_paths = []
    all_drifts = []
    all_probs = []

    for i in range(num_simulations):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Generated simulation {i + 1}/{num_simulations}")

        # Generate path with unique random seed and optimized parameters
        try:
            path = []
            drift = []
            prob = []
            
            current_price = engine.S_0
            
            for j, exposure in enumerate(interpolated_exposure):
                # Map exposure to probability with optimized parameters
                probability = engine.exposure_to_probability(exposure)
                prob.append(probability)
                
                # Map probability to drift with optimized parameters
                mu = engine.compute_mu_from_probability(probability)
                drift.append(mu)
                
                # Generate price step for 1 second with the drift
                dt = 1.0 / (365 * 24 * 3600)  # 1 second in years
                np.random.seed(random_seed + i + j)
                dW = np.random.normal(0, np.sqrt(dt))
                
                # Apply geometric Brownian motion step
                current_price = current_price * np.exp((mu - 0.5 * engine.sigma_base**2) * dt + engine.sigma_base * dW)
                path.append(current_price)
            
            all_paths.append(path)
            
            # Save the first simulation's details for analysis
            if i == 0:
                all_drifts = drift
                all_probs = prob
                
        except Exception as e:
            print(f"Error in simulation {i+1}: {e}")
            continue

    if not all_paths:
        print("No successful simulations generated!")
        return

    # Convert to numpy array for easier manipulation
    all_paths = np.array(all_paths)

    # Calculate mean path and confidence bands
    mean_path = np.mean(all_paths, axis=0)
    lower_band = np.percentile(all_paths, 10, axis=0)
    upper_band = np.percentile(all_paths, 90, axis=0)

    # Create single plot visualization - focus only on the index
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    
    # Plot individual paths
    print("  Plotting individual paths...")
    for i in range(min(15, len(all_paths))):  # Plot up to 15 individual paths
        ax.plot(time_seconds, all_paths[i], alpha=0.15, color="steelblue", linewidth=0.8)

    # Plot mean and bands
    print("  Plotting mean paths...")
    ax.plot(time_seconds, mean_path, color="darkblue", linewidth=2.5, label="Mean Index Path (Optimized)")
    ax.fill_between(
        time_seconds,
        lower_band,
        upper_band,
        color="blue",
        alpha=0.2,
        label="80% Confidence Band",
    )

    # Add horizontal line at starting price
    ax.axhline(
        y=engine.S_0, color="red", linestyle="--", alpha=0.7, label="Starting Price"
    )

    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Index Price", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_title(
        f"Risk-Optimized Supply-Demand Index: Per-Second Dynamic Response to Vol 75 Exposure\n"
        f"({len(all_paths)} simulations, {len(time_seconds):,} data points)",
        fontsize=14,
    )

    # Format x-axis to show time in hours
    ax_hours = ax.twiny()
    ax_hours.set_xlim(ax.get_xlim())
    hour_ticks = np.arange(0, total_duration_seconds, 3600)  # Every hour
    ax_hours.set_xticks(hour_ticks)
    ax_hours.set_xticklabels([f"{int(h/3600)}h" for h in hour_ticks])
    ax_hours.set_xlabel("Time (hours)", fontsize=10)

    # Create plots directory if it doesn't exist
    plots_dir = "plots/risk"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/6_dynamic_index_path.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Dynamic index path plot saved to {plots_dir}/6_dynamic_index_path.png")
    print(f"Generated {len(time_seconds):,} per-second index values")
    print(f"Exposure correlation analysis:")
    
    # Calculate correlation between mean index movement and exposure
    index_returns = np.diff(mean_path) / mean_path[:-1]  # Calculate returns
    exposure_changes = np.diff(interpolated_exposure)
    
    if len(index_returns) == len(exposure_changes):
        correlation = np.corrcoef(index_returns, exposure_changes)[0, 1]
        print(f"  Correlation between index returns and exposure changes: {correlation:.3f}")
        
        # Analyze directional alignment
        positive_exposure_periods = exposure_changes > 0
        positive_index_periods = index_returns > 0
        alignment = np.mean(positive_exposure_periods == positive_index_periods)
        print(f"  Directional alignment (same direction): {alignment:.1%}")


def create_comprehensive_risk_visualization(
    engine: RiskOptimizedSupplyDemandIndexEngine,
    backtest_results: Dict[str, Any],
    exposure_data: pd.Series,
    sample_size: int = 20
) -> None:
    """
    Create comprehensive visualization of risk optimization results
    
    Parameters:
    -----------
    engine : RiskOptimizedSupplyDemandIndexEngine
        Optimized engine instance
    backtest_results : Dict[str, Any]
        Results from comprehensive backtesting
    exposure_data : pd.Series
        Original exposure data
    sample_size : int
        Number of exposure scenarios to visualize
    """
    print("Creating comprehensive risk optimization visualization...")
    
    # Create plots directory
    plots_dir = "plots/risk"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Sample exposure data for visualization
    np.random.seed(42)
    if len(exposure_data) > sample_size:
        sampled_indices = np.random.choice(len(exposure_data), size=sample_size, replace=False)
        sampled_exposures = exposure_data.iloc[sampled_indices].values
    else:
        sampled_exposures = exposure_data.values[:sample_size]
    
    # Apply optimal parameters to engine
    optimal_params = backtest_results['optimal_params']
    for key, value in optimal_params.items():
        if hasattr(engine, f"{key}_base"):
            setattr(engine, f"{key}_base", value)
        else:
            setattr(engine, key, value)
    
    # Process exposure scenarios with optimized parameters
    results = engine.process_exposure_data(
        exposure_data=sampled_exposures.tolist(),
        duration_in_seconds=3600,  # 1 hour
        num_paths_per_exposure=100,
        random_seed=42
    )
    
    # Create comprehensive visualization
    engine.create_comprehensive_visualization(
        results=results,
        max_paths_to_plot=30,
        figsize=(20, 16),
        save_path=f"{plots_dir}/8_combined_exposure_index.png"
    )
    
    print(f"Comprehensive visualization saved to {plots_dir}/8_combined_exposure_index.png")


def generate_risk_optimization_report(
    engine: RiskOptimizedSupplyDemandIndexEngine,
    backtest_results: Dict[str, Any],
    exposure_data: pd.Series
) -> str:
    """
    Generate comprehensive risk optimization report
    
    Parameters:
    -----------
    engine : RiskOptimizedSupplyDemandIndexEngine
        Engine instance
    backtest_results : Dict[str, Any]
        Backtest results
    exposure_data : pd.Series
        Original exposure data
        
    Returns:
    --------
    str
        Formatted report
    """
    report = []
    report.append("=" * 80)
    report.append("RISK-OPTIMIZED VOL 75 SUPPLY-DEMAND INDEX ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Data Summary
    report.append("DATA SUMMARY")
    report.append("-" * 40)
    report.append(f"Total exposure points: {len(exposure_data):,}")
    report.append(f"Exposure range: {exposure_data.min():,.0f} to {exposure_data.max():,.0f}")
    report.append(f"Mean exposure: {exposure_data.mean():,.0f}")
    report.append(f"Exposure std dev: {exposure_data.std():,.0f}")
    report.append("")
    
    # Engine Configuration
    report.append("ENGINE CONFIGURATION")
    report.append("-" * 40)
    report.append(f"Base volatility (σ): {engine.sigma_base:.1%}")
    report.append(f"Exposure scale: {engine.scale_base:,.0f}")
    report.append(f"Probability range (k): ±{engine.k_base:.1%}")
    report.append(f"Time horizon: {engine.T * 365 * 24:.1f} hours")
    report.append(f"Starting price: {engine.S_0:,.0f}")
    report.append(f"Noise injection level: {engine.noise_injection_level:.1%}")
    report.append(f"Max daily movement: {engine.max_daily_movement:.1%}")
    report.append("")
    
    # Optimization Results
    baseline = backtest_results['baseline_metrics']
    optimized = backtest_results['optimized_metrics']
    improvement = backtest_results['improvement']
    
    report.append("OPTIMIZATION RESULTS")
    report.append("-" * 40)
    report.append(f"{'Metric':<20} {'Baseline':<12} {'Optimized':<12} {'Improvement':<12}")
    report.append("-" * 60)
    report.append(f"{'Responsiveness':<20} {baseline['responsiveness']:<12.3f} {optimized['responsiveness']:<12.3f} {improvement['responsiveness']:<12.3f}")
    report.append(f"{'Exploitability':<20} {baseline['exploitability']:<12.3f} {optimized['exploitability']:<12.3f} {improvement['exploitability']:<12.3f}")
    report.append(f"{'Risk Score':<20} {baseline['risk']:<12.3f} {optimized['risk']:<12.3f} {improvement['risk']:<12.3f}")
    report.append(f"{'Unpredictability':<20} {baseline['unpredictability']:<12.3f} {optimized['unpredictability']:<12.3f} {improvement['unpredictability']:<12.3f}")
    report.append("")
    
    # Optimal Parameters
    optimal_params = backtest_results['optimal_params']
    report.append("OPTIMAL PARAMETERS")
    report.append("-" * 40)
    for param, value in optimal_params.items():
        if isinstance(value, float):
            if param in ['sigma', 'k', 'psychology_bull_factor', 'psychology_bear_factor', 'noise_injection_level']:
                report.append(f"{param:<25}: {value:.1%}")
            else:
                report.append(f"{param:<25}: {value:,.2f}")
        else:
            report.append(f"{param:<25}: {value}")
    report.append("")
    
    # Risk Assessment
    report.append("RISK ASSESSMENT")
    report.append("-" * 40)
    
    # Calculate risk metrics
    overall_score = (
        improvement['responsiveness'] * 0.3 +
        improvement['exploitability'] * 0.4 +
        improvement['risk'] * 0.2 +
        improvement['unpredictability'] * 0.1
    )
    
    if overall_score > 0.1:
        risk_level = "LOW"
        risk_color = "✓"
    elif overall_score > 0:
        risk_level = "MEDIUM"
        risk_color = "~"
    else:
        risk_level = "HIGH"
        risk_color = "⚠"
    
    report.append(f"Overall optimization score: {overall_score:.3f}")
    report.append(f"Risk level: {risk_color} {risk_level}")
    report.append("")
    
    # Exploitation Resistance
    exploitation = backtest_results['exploitation_results']
    report.append("EXPLOITATION RESISTANCE")
    report.append("-" * 40)
    for test_name, results in exploitation.items():
        if 'error' not in results:
            status = results.get('status', 'unknown')
            score = results.get('resistance_score', 0)
            report.append(f"{test_name:<25}: {status:<15} (Score: {score:.3f})")
        else:
            report.append(f"{test_name:<25}: ERROR")
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 40)
    
    if improvement['responsiveness'] > 0.05:
        report.append("✓ Good responsiveness to market exposure achieved")
    else:
        report.append("⚠ Consider increasing responsiveness parameters")
    
    if improvement['exploitability'] > 0.05:
        report.append("✓ Strong anti-exploitation measures in place")
    else:
        report.append("⚠ Review anti-exploitation mechanisms")
    
    if improvement['risk'] > 0.05:
        report.append("✓ Effective risk reduction achieved")
    else:
        report.append("⚠ Monitor directional bias risk closely")
    
    # Parameter recommendations
    report.append("")
    report.append("PARAMETER RECOMMENDATIONS:")
    
    if optimal_params.get('sigma', 0) > 0.5:
        report.append("• Consider reducing volatility parameter for stability")
    
    if optimal_params.get('noise_injection_level', 0) < 0.02:
        report.append("• Consider increasing noise injection for better anti-exploitation")
    
    if optimal_params.get('scale', 0) < 100000:
        report.append("• Scale parameter may be too sensitive - monitor for over-reaction")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """
    Main function to run the complete risk-optimized Vol 75 analysis
    """
    print("Starting Risk-Optimized Vol 75 Supply-Demand Index Analysis...")
    print("=" * 60)
    
    try:
        # Step 1: Load trade data
        print("Step 1: Loading Vol 75 trade data...")
        trades_df = load_vol75_trade_data()
        
        # Step 2: Calculate net exposure
        print("\nStep 2: Calculating net exposure...")
        exposure_data = calculate_net_exposure(trades_df, window_minutes=5)
        
        # Step 3: Run risk-optimized analysis
        print("\nStep 3: Running risk-optimized analysis...")
        engine, backtest_results = run_risk_optimized_analysis(
            exposure_data=exposure_data,
            sample_size=50,
            random_seed=1505,
            optimization_method='scipy',  # Use scipy for reliability
            n_optimization_calls=25
        )
        
        # Step 4: Generate optimization report
        print("\nStep 4: Generating optimization report...")
        report = generate_risk_optimization_report(engine, backtest_results, exposure_data)
        print(report)
        
        # Save report to file
        with open('risk_vol75_optimization_report.txt', 'w') as f:
            f.write(report)
        print("\nReport saved to 'risk_vol75_optimization_report.txt'")
        
        # Step 5: Create comprehensive visualization
        print("\nStep 5: Creating comprehensive visualization...")
        create_comprehensive_risk_visualization(
            engine=engine,
            backtest_results=backtest_results,
            exposure_data=exposure_data,
            sample_size=20
        )
        
        # Step 6: Generate dynamic index paths with optimized parameters
        print("\nStep 6: Generating dynamic index paths...")
        # Use ALL exposure data for path generation (full 1440 points)
        print(f"Using all {len(exposure_data)} exposure points for dynamic path generation")
        generate_dynamic_index_path_optimized(
            engine=engine,
            exposure_series=exposure_data,  # Use full dataset
            num_simulations=30,
            random_seed=1505,
            ma_window=12,
            seconds_per_point=1  # One point every second for high-resolution index
        )
        
        # Step 7: Save results
        print("\nStep 7: Saving results...")
        results_data = {
            'backtest_results': backtest_results,
            'engine_params': {
                'sigma_base': engine.sigma_base,
                'scale_base': engine.scale_base,
                'k_base': engine.k_base,
                'noise_injection_level': engine.noise_injection_level,
                'max_daily_movement': engine.max_daily_movement
            },
            'data_summary': {
                'total_trades': len(trades_df),
                'total_exposure_points': len(exposure_data),
                'exposure_range': [float(exposure_data.min()), float(exposure_data.max())],
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
        with open('risk_vol75_analysis_results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print("Analysis complete! Results saved to:")
        print("  - risk_vol75_optimization_report.txt")
        print("  - risk_vol75_analysis_results.json")
        print("  - plots/risk/ (visualization files)")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
