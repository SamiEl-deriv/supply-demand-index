"""
Vol 75 Supply-Demand Index Analysis

This script analyzes Vol 75 trade data using the supply-demand index engine
with advanced mathematical models including:
1. Smooth, non-linear exposure-to-probability mapping
2. Adaptive drift calculation with market regime awareness
3. Sophisticated stochastic process with mean reversion and fractal characteristics

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
from adjusted_supply_demand_index_engine import SupplyDemandIndexEngine

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


def run_analysis(
    exposure_data: pd.Series, sample_size: int = 50, random_seed: int = 1505
) -> Tuple[SupplyDemandIndexEngine, Dict[str, Any]]:
    """
    Run supply-demand index analysis on exposure data

    Parameters:
    -----------
    exposure_data : pd.Series
        Series of net exposure values
    sample_size : int
        Number of exposure points to sample for detailed analysis
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    Tuple[SupplyDemandIndexEngine, Dict[str, Any]]
        Engine instance and results dictionary
    """
    # Initialize the engine with optimized parameters
    engine = SupplyDemandIndexEngine(
        sigma=0.3,  # 30% volatility - typical for crypto assets
        scale=150_000,  # Exposure sensitivity calibrated to Vol 75
        k=0.4,  # Max probability deviation (0.1 to 0.9 range)
        T=1.0 / (365 * 24),  # 1 hour time horizon
        S_0=100_000,  # Starting price (100,000)
        smoothness_factor=2.0,  # Controls smoothness of transitions
    )

    print("Initialized Enhanced Supply-Demand Index Engine with parameters:")
    print(f"  sigma: {engine.sigma:.1%}")
    print(f"  scale: {engine.scale}")
    print(f"  k: {engine.k}")
    print(f"  T: {engine.T * 365 * 24:.1f} hours")
    print(f"  S_0: {engine.S_0}")
    print(f"  smoothness_factor: {engine.smoothness_factor}")

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
        f"Generating supply-demand fGBM index from {len(exposure_data)} exposure points..."
    )
    print(f"Sampled {len(sampled_exposure)} exposure points for analysis")
    print(
        f"Exposure range: {min(sampled_exposure):,.0f} to {max(sampled_exposure):,.0f}"
    )

    # Process the exposure data
    results = engine.process_exposure_data(
        exposure_data=sampled_exposure,
        duration_in_seconds=3600,  # 1 hour simulation
        num_paths_per_exposure=50,  # 50 Monte Carlo paths per exposure
        random_seed=random_seed,
    )

    return engine, results


def generate_dynamic_index_path(
    engine: SupplyDemandIndexEngine,
    exposure_series: pd.Series,
    num_simulations: int = 100,
    random_seed: int = 1505,
    ma_window: int = 12,  # 1 hour with 5min data (12 points)
    seconds_per_point: int = 1,  # Generate a point every second
) -> None:
    """
    Generate dynamic index paths that respond to time-varying exposure

    Parameters:
    -----------
    engine : SupplyDemandIndexEngine
        Initialized engine instance
    exposure_series : pd.Series
        Time series of exposure values
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
        f"Generating dynamic index paths that respond to time-varying exposure with {ma_window}-point weighted moving average..."
    )
    print(f"Using random seed base: {random_seed} for dynamic path generation")
    print(f"Generating index points every {seconds_per_point} second(s)")

    # Interpolate exposure data to per-second resolution
    # First, create a continuous time index
    original_index = np.arange(len(exposure_series))
    # Create a new index with second-level resolution
    seconds_per_minute = 60
    new_index = np.linspace(
        0,
        len(exposure_series) - 1,
        len(exposure_series) * seconds_per_minute // seconds_per_point,
    )

    # Interpolate the exposure values to the new index
    interpolated_exposure = np.interp(new_index, original_index, exposure_series.values)

    # Create a new Series with the interpolated values
    interpolated_exposure_series = pd.Series(interpolated_exposure)

    print(
        f"Interpolated exposure data from {len(exposure_series)} points to {len(interpolated_exposure_series)} points"
    )

    # Run multiple simulations
    all_paths = []
    all_drifts = []
    all_smoothed_drifts = []
    all_probs = []

    for i in range(num_simulations):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Generated simulation {i + 1}/{num_simulations}")

        # Generate path with unique random seed
        path, drift, smoothed_drift, prob = engine.generate_dynamic_exposure_path(
            exposure_series=interpolated_exposure_series,
            random_seed=random_seed + i,
            ma_window=ma_window,
        )
        all_paths.append(path)

        # Save the first simulation's details for the combined plot
        if i == 0:
            all_drifts = drift
            all_smoothed_drifts = smoothed_drift
            all_probs = prob

    # Convert to numpy array for easier manipulation
    all_paths = np.array(all_paths)

    # Calculate mean path
    mean_path = np.mean(all_paths, axis=0)

    # Calculate percentiles for confidence bands
    lower_band = np.percentile(all_paths, 10, axis=0)
    upper_band = np.percentile(all_paths, 90, axis=0)

    # Create visualization
    plt.figure(figsize=(16, 10))

    # Plot individual paths
    print("  Plotting individual paths...")
    for i in range(min(20, num_simulations)):  # Plot up to 20 individual paths
        plt.plot(all_paths[i], alpha=0.15, color="steelblue", linewidth=0.8)

    # Plot mean and bands
    print("  Plotting mean paths...")
    plt.plot(mean_path, color="darkblue", linewidth=2.5, label="Mean Path")
    plt.fill_between(
        range(len(mean_path)),
        lower_band,
        upper_band,
        color="blue",
        alpha=0.2,
        label="80% Confidence Band",
    )

    # Add horizontal line at starting price
    plt.axhline(
        y=engine.S_0, color="red", linestyle="--", alpha=0.7, label="Starting Price"
    )

    # Formatting
    plt.title(
        f"Supply-Demand Index: Dynamic Response to Vol 75 Exposure\n"
        f"({num_simulations} simulations, {ma_window}-point weighted moving average, {seconds_per_point}-second resolution)",
        fontsize=14,
    )
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Index Price", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # Create plots directory if it doesn't exist
    plots_dir = "plots/adjusted"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created directory: {plots_dir}")

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/6_dynamic_index_path.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Create a second plot showing the exposure data that generated the paths
    plt.figure(figsize=(16, 6))
    plt.plot(interpolated_exposure, color="darkgreen", linewidth=1.5)
    plt.title("Vol 75 Net Exposure Over Time", fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Net Exposure ($)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/7_exposure_data.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Create a combined plot with exposure and index on the same time axis
    create_combined_exposure_index_plot(
        exposure=interpolated_exposure, index_path=mean_path, plots_dir=plots_dir
    )

    print(f"Created 3 additional plots in the '{plots_dir}' directory")


def create_combined_exposure_index_plot(
    exposure: np.ndarray, index_path: np.ndarray, plots_dir: str = "plots"
) -> None:
    """
    Create a combined plot showing both exposure and index on the same time axis

    Parameters:
    -----------
    exposure : np.ndarray
        Array of exposure values
    index_path : np.ndarray
        Array of index values
    plots_dir : str
        Directory to save the plot
    """
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(16, 8))

    # Plot exposure on the first y-axis
    color = "tab:green"
    ax1.set_xlabel("Time (seconds)", fontsize=12)
    ax1.set_ylabel("Net Exposure ($)", color=color, fontsize=12)
    ax1.plot(exposure, color=color, linewidth=1.5, label="Net Exposure")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Create second y-axis for the index
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Supply-Demand Index", color=color, fontsize=12)

    # Ensure the index path has the same length as the exposure
    # If not, we need to interpolate or truncate
    if len(index_path) != len(exposure):
        # Interpolate index_path to match exposure length
        x_orig = np.linspace(0, 1, len(index_path))
        x_new = np.linspace(0, 1, len(exposure))
        index_path = np.interp(x_new, x_orig, index_path)

    ax2.plot(index_path, color=color, linewidth=2.0, label="Supply-Demand Index")
    ax2.tick_params(axis="y", labelcolor=color)

    # Add title and legend
    plt.title("Combined Exposure and Supply-Demand Index", fontsize=14)

    # Create a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=12)

    # Save the figure
    plt.tight_layout()
    plt.savefig(
        f"{plots_dir}/8_combined_exposure_index.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_separate_visualizations(
    engine: SupplyDemandIndexEngine, results: Dict[str, Any]
) -> None:
    """
    Create separate, clear visualizations for each aspect of the analysis

    Parameters:
    -----------
    engine : SupplyDemandIndexEngine
        Initialized engine instance
    results : Dict[str, Any]
        Results dictionary from process_exposure_data
    """
    # Create plots directory if it doesn't exist
    plots_dir = "plots/adjusted"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created directory: {plots_dir}")

    # 1. Exposure vs Probability Mapping
    plt.figure(figsize=(12, 8))
    exposures = results["exposures"]
    probabilities = results["probabilities"]

    # Create smooth theoretical curve
    exposure_range = np.linspace(min(exposures) * 1.2, max(exposures) * 1.2, 1000)
    theoretical_probs = [engine.exposure_to_probability(exp) for exp in exposure_range]

    # Plot theoretical curve
    plt.plot(
        exposure_range,
        theoretical_probs,
        "-",
        color="blue",
        linewidth=2.5,
        label="Theoretical Mapping",
    )

    # Plot actual data points
    plt.scatter(
        exposures, probabilities, color="red", s=50, alpha=0.8, label="Data Points"
    )

    # Add reference lines
    plt.axhline(y=0.5, color="black", linestyle="--", alpha=0.6, label="Neutral (0.5)")
    plt.axvline(x=0, color="black", linestyle=":", alpha=0.6)

    # Formatting
    plt.title("Exposure-to-Probability Mapping", fontsize=16)
    plt.xlabel("Net Exposure ($)", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(
        f"{plots_dir}/1_exposure_probability_mapping.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 2. Probability vs Drift Mapping
    plt.figure(figsize=(12, 8))
    mu_values = results["mu_values"]

    # Create theoretical curve
    prob_range = np.linspace(0.1, 0.9, 1000)
    theoretical_mu = [engine.compute_mu_from_probability(p) for p in prob_range]

    # Plot theoretical curve
    plt.plot(
        prob_range,
        theoretical_mu,
        "-",
        color="blue",
        linewidth=2.5,
        label="Theoretical Mapping",
    )

    # Plot actual data points
    plt.scatter(
        probabilities, mu_values, color="red", s=50, alpha=0.8, label="Data Points"
    )

    # Add reference lines
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.6, label="Zero Drift")
    plt.axvline(
        x=0.5, color="black", linestyle=":", alpha=0.6, label="Neutral Probability"
    )

    # Formatting
    plt.title("Probability-to-Drift Mapping", fontsize=16)
    plt.xlabel("Probability", fontsize=14)
    plt.ylabel("Drift (μ)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(
        f"{plots_dir}/2_probability_drift_mapping.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 3. Sample Price Paths
    # Select a few representative scenarios
    n_scenarios = min(6, len(results["exposures"]))
    indices = np.linspace(0, len(results["exposures"]) - 1, n_scenarios, dtype=int)

    for i, idx in enumerate(indices):
        plt.figure(figsize=(10, 6))

        exposure = results["exposures"][idx]
        paths = results["price_paths"][idx]
        mean_path = results["mean_paths"][idx]

        # Plot sample paths
        for j, path in enumerate(paths[:20]):  # Show first 20 paths
            plt.plot(path, color="lightblue", alpha=0.3, linewidth=0.5)

        # Plot mean path
        plt.plot(mean_path, color="red", linewidth=2, label="Mean Path")
        plt.axhline(
            y=engine.S_0,
            color="black",
            linestyle="--",
            alpha=0.5,
            label="Initial Price",
        )

        # Formatting
        plt.title(f"Price Paths for Exposure: ${exposure:,.0f}", fontsize=14)
        plt.xlabel("Time Steps", fontsize=12)
        plt.ylabel("Price", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(
            f"{plots_dir}/3_{i + 1}_price_paths_exp_{int(exposure / 1000)}k.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # 4. Validation Results
    plt.figure(figsize=(12, 8))

    # Extract validation rates
    summary_stats = results["summary_stats"]
    exposures_plot = [s["exposure"] for s in summary_stats]
    mu_rates = [s["mu_validation_rate"] for s in summary_stats]

    # Sort by exposure
    sorted_indices = np.argsort(exposures_plot)
    exposures_plot = [exposures_plot[i] for i in sorted_indices]
    mu_rates = [mu_rates[i] for i in sorted_indices]

    # Plot validation rates
    plt.bar(
        range(len(exposures_plot)),
        mu_rates,
        color="purple",
        alpha=0.7,
        label="μ Validation Rate",
    )

    # Add reference line
    plt.axhline(y=0.5, color="black", linestyle="--", alpha=0.6, label="50% Threshold")

    # Formatting
    plt.title("Drift Validation Results by Exposure Level", fontsize=16)
    plt.xlabel("Exposure Scenarios (sorted by amount)", fontsize=14)
    plt.ylabel("Validation Rate", fontsize=14)
    plt.xticks([])  # Hide x-ticks for clarity
    plt.grid(True, alpha=0.3, axis="y")
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/4_validation_results.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Removed fractal characteristics plot since we eliminated fractal dimension
    # to prevent predictable patterns that could be exploited

    print(f"Created 4 separate visualization plots in the '{plots_dir}' directory")


def print_analysis_summary(
    trades_df: pd.DataFrame,
    exposure_data: pd.Series,
    engine: SupplyDemandIndexEngine,
    results: Dict[str, Any],
) -> None:
    """
    Print a comprehensive summary of the analysis results

    Parameters:
    -----------
    trades_df : pd.DataFrame
        DataFrame containing trade data
    exposure_data : pd.Series
        Series of net exposure values
    engine : SupplyDemandIndexEngine
        Engine instance used for analysis
    results : Dict[str, Any]
        Results dictionary from process_exposure_data
    """
    # Calculate total volume
    total_volume = trades_df["amount"].sum()

    # Calculate exposure statistics
    exposure_std = exposure_data.std()

    print("\n" + "=" * 60)
    print("VOL 75 SUPPLY-DEMAND INDEX ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Data Period: {exposure_data.index[0]} to {exposure_data.index[-1]}")
    print(f"Total Trades: {len(trades_df):,}")
    print(f"Total Volume: ${total_volume:,.0f}")
    print(f"Exposure Range: ${exposure_data.min():,.0f} to ${exposure_data.max():,.0f}")
    print(f"Exposure Std Dev: ${exposure_std:,.0f}")

    # Print validation summary
    engine.print_validation_summary(results)

    # Generate and print detailed report
    report = engine.generate_summary_report(results)
    print("\nDetailed Analysis Report:")
    print(report.to_string())


def main():
    """Main function to run the Vol 75 supply-demand index analysis"""
    print("Starting Vol 75 Supply-Demand Index Analysis...")

    # Load Vol 75 trade data
    print("Loading Vol 75 trade data...")
    trades_df = load_vol75_trade_data()

    # Calculate net exposure in 5-minute windows
    exposure_data = calculate_net_exposure(trades_df, window_minutes=5)

    # Run analysis
    engine, results = run_analysis(
        exposure_data=exposure_data, sample_size=50, random_seed=2024
    )

    # Create separate visualizations for clarity
    create_separate_visualizations(engine, results)

    # Generate dynamic index path
    generate_dynamic_index_path(
        engine=engine,
        exposure_series=exposure_data,
        num_simulations=20,
        random_seed=2024,
        ma_window=12,  # 1 hour with 5min data
    )

    # Print analysis summary
    print_analysis_summary(trades_df, exposure_data, engine, results)

    print("\nAnalysis complete! Check generated PNG files for visualizations.")


if __name__ == "__main__":
    main()
