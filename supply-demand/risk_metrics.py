"""
Risk Metrics Analysis for Supply-Demand Index
Comprehensive risk testing and parameter optimization

This module provides:
1. Complete risk assessment framework
2. Parameter optimization with constraints
3. All plots from risk vol 75 analysis
4. Parameter classification (fixed vs adjustable)
5. Real-time risk monitoring capabilities

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
from scipy.optimize import minimize
from engine import SupplyDemandIndexEngine

# Set plot style
plt.style.use("ggplot")
plt.rcParams.update({'font.size': 11})


def load_vol75_trade_data(
    data_dir: str = "/Users/samielmokh/Downloads/data/trade/vol_75",
) -> pd.DataFrame:
    """Load Vol 75 trade data from CSV files"""
    csv_files = glob.glob(f"{data_dir}/*.csv")
    csv_files.sort()
    print(f"Found {len(csv_files)} CSV files")

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

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df["time_deal"] = pd.to_datetime(combined_df["time_deal"], format="mixed")
    combined_df = combined_df.sort_values("time_deal").reset_index(drop=True)

    print(f"Loaded {len(combined_df):,} total trades")
    print(f"Date range: {combined_df['time_deal'].min()} to {combined_df['time_deal'].max()}")

    trades_df = pd.DataFrame()
    trades_df["timestamp"] = combined_df["time_deal"]
    trades_df["direction"] = combined_df["action"].map({0: "buy", 1: "sell"})
    trades_df["amount"] = combined_df["volume_usd"]

    return trades_df


def calculate_net_exposure(trades_df: pd.DataFrame, window_minutes: int = 5) -> pd.Series:
    """Calculate net exposure in fixed time windows"""
    print(f"Calculating net exposure with {window_minutes}min windows...")

    trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"])
    start_time = trades_df["timestamp"].min().floor("D")
    end_time = trades_df["timestamp"].max().ceil("D")
    
    windows = pd.date_range(start=start_time, end=end_time, freq=f"{window_minutes}min")
    exposure = pd.Series(0.0, index=windows[:-1])

    for i in range(len(windows) - 1):
        window_start = windows[i]
        window_end = windows[i + 1]
        mask = (trades_df["timestamp"] >= window_start) & (trades_df["timestamp"] < window_end)
        window_trades = trades_df[mask]

        if len(window_trades) > 0:
            buys = window_trades[window_trades["direction"] == "buy"]["amount"].sum()
            sells = window_trades[window_trades["direction"] == "sell"]["amount"].sum()
            net = buys - sells
            exposure[window_start] = net

    exposure = exposure[exposure != 0]
    print(f"Generated {len(exposure)} time windows with trades")
    print(f"Exposure range: {exposure.min():,.0f} to {exposure.max():,.0f}")

    return exposure


def optimize_parameters(
    exposure_data: pd.Series, 
    sample_size: int = 50,
    n_iterations: int = 30
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Optimize the 4 key parameters: sigma, scale, k, smoothness_factor
    
    Returns:
    --------
    Tuple[Dict[str, float], Dict[str, Any]]
        Optimal parameters and detailed results
    """
    print("Optimizing 4 key parameters...")
    print("Parameters to optimize:")
    print("  1. sigma (Base volatility)")
    print("  2. scale (Exposure sensitivity)")
    print("  3. k (Max probability deviation)")
    print("  4. smoothness_factor (Transition smoothness)")
    print()
    
    # Create baseline engine
    engine = SupplyDemandIndexEngine()
    
    # Sample exposure data
    np.random.seed(1505)
    if len(exposure_data) > sample_size:
        sampled_indices = np.random.choice(len(exposure_data), size=sample_size, replace=False)
        sampled_exposure = exposure_data.iloc[sampled_indices].values
    else:
        sampled_exposure = exposure_data.values

    print(f"Using {len(sampled_exposure)} exposure points for optimization")
    print(f"Exposure range: {min(sampled_exposure):,.0f} to {max(sampled_exposure):,.0f}")

    # Define parameter bounds
    bounds = [
        (0.10, 0.60),     # sigma: 10% to 60%
        (50000, 300000),  # scale: 50k to 300k
        (0.20, 0.50),     # k: 20% to 50%
        (1.0, 4.0),       # smoothness_factor: 1.0 to 4.0
    ]

    def objective(params):
        """Objective function focusing on the 4 key parameters"""
        sigma, scale, k, smoothness = params
        
        # Store original parameters
        original_params = {
            'sigma': engine.sigma,
            'scale': engine.scale,
            'k': engine.k,
            'smoothness_factor': engine.smoothness_factor
        }
        
        # Set new parameters
        engine.sigma = sigma
        engine.scale = scale
        engine.k = k
        engine.smoothness_factor = smoothness
        
        try:
            # Run simulation with these parameters
            results = engine.process_exposure_data(
                exposure_data=sampled_exposure.tolist(),
                duration_in_seconds=3600,  # 1 hour
                num_paths_per_exposure=50,  # Reduced for speed
                random_seed=42
            )
            
            # Calculate performance metrics
            summary_stats = results['summary_stats']
            
            # Calculate responsiveness (correlation between exposure and probability)
            exposures = [s['exposure'] for s in summary_stats]
            probabilities = [s['probability'] for s in summary_stats]
            if len(exposures) > 1:
                correlation = np.corrcoef(exposures, probabilities)[0, 1]
                responsiveness = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                responsiveness = 0.0
            
            # Calculate validation rate (how often drift predictions are correct)
            validation_rate = np.mean([s['mu_validation_rate'] for s in summary_stats])
            
            # Calculate price stability (lower volatility of final prices is better)
            final_prices = []
            for paths in results['price_paths']:
                for path in paths:
                    final_prices.append(path[-1])
            price_volatility = np.std(final_prices) / np.mean(final_prices) if final_prices else 1.0
            
            # Calculate smoothness (lower variation in probabilities is smoother)
            prob_variation = np.std(probabilities) if len(probabilities) > 1 else 0.0
            
            # Weighted objective (minimize)
            # We want: high responsiveness, high validation rate, low price volatility, moderate smoothness
            objective_value = (
                -0.30 * responsiveness +          # Want high responsiveness
                -0.30 * validation_rate +         # Want high validation rate
                0.25 * price_volatility +         # Want low price volatility
                0.15 * min(prob_variation, 0.2)   # Want moderate smoothness (not too rigid)
            )
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            objective_value = 1.0  # High penalty
        
        # Restore original parameters
        for key, value in original_params.items():
            setattr(engine, key, value)
        
        return objective_value

    # Initial guess (current values)
    x0 = [
        engine.sigma,
        engine.scale,
        engine.k,
        engine.smoothness_factor
    ]

    print(f"Starting optimization with {n_iterations} evaluations...")
    print("Initial parameters:")
    print(f"  sigma: {x0[0]:.1%}")
    print(f"  scale: {x0[1]:,.0f}")
    print(f"  k: {x0[2]:.1%}")
    print(f"  smoothness_factor: {x0[3]:.1f}")
    print()

    # Run optimization
    result = minimize(
        fun=objective,
        x0=x0,
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': n_iterations}
    )

    # Extract optimal parameters
    optimal_params = {
        'sigma': result.x[0],
        'scale': result.x[1],
        'k': result.x[2],
        'smoothness_factor': result.x[3]
    }

    print("Optimization completed!")
    print(f"Best objective value: {result.fun:.4f}")
    print("Optimal parameters:")
    print(f"  sigma: {optimal_params['sigma']:.1%}")
    print(f"  scale: {optimal_params['scale']:,.0f}")
    print(f"  k: {optimal_params['k']:.1%}")
    print(f"  smoothness_factor: {optimal_params['smoothness_factor']:.1f}")
    print()

    # Test optimal parameters
    print("Testing optimal parameters...")
    
    # Apply optimal parameters
    engine.sigma = optimal_params['sigma']
    engine.scale = optimal_params['scale']
    engine.k = optimal_params['k']
    engine.smoothness_factor = optimal_params['smoothness_factor']
    
    # Get baseline metrics (with original parameters)
    engine_baseline = SupplyDemandIndexEngine()
    baseline_results = engine_baseline.process_exposure_data(
        exposure_data=sampled_exposure.tolist(),
        duration_in_seconds=3600,
        num_paths_per_exposure=50,
        random_seed=42
    )
    
    # Get optimized metrics
    optimized_results = engine.process_exposure_data(
        exposure_data=sampled_exposure.tolist(),
        duration_in_seconds=3600,
        num_paths_per_exposure=50,
        random_seed=42
    )
    
    # Calculate improvements
    baseline_validation = np.mean([s['mu_validation_rate'] for s in baseline_results['summary_stats']])
    optimized_validation = np.mean([s['mu_validation_rate'] for s in optimized_results['summary_stats']])
    
    results = {
        'optimal_params': optimal_params,
        'baseline_validation_rate': baseline_validation,
        'optimized_validation_rate': optimized_validation,
        'validation_improvement': optimized_validation - baseline_validation,
        'optimization_success': result.success,
        'objective_value': result.fun,
        'n_evaluations': result.nfev if hasattr(result, 'nfev') else n_iterations
    }

    return optimal_params, results


def create_all_risk_plots(
    engine: SupplyDemandIndexEngine,
    exposure_data: pd.Series,
    optimal_params: Dict[str, float],
    sample_size: int = 20
) -> None:
    """
    Create all risk analysis plots from the original risk vol 75 analysis
    """
    print("Creating comprehensive risk analysis plots...")
    
    # Create plots directory
    plots_dir = "plots/risk_metrics"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Apply optimal parameters
    engine.sigma = optimal_params['sigma']
    engine.scale = optimal_params['scale']
    engine.k = optimal_params['k']
    engine.smoothness_factor = optimal_params['smoothness_factor']
    
    # Sample exposure data for visualization
    np.random.seed(42)
    if len(exposure_data) > sample_size:
        sampled_indices = np.random.choice(len(exposure_data), size=sample_size, replace=False)
        sampled_exposures = exposure_data.iloc[sampled_indices].values
    else:
        sampled_exposures = exposure_data.values[:sample_size]
    
    # Process exposure scenarios with optimized parameters
    results = engine.process_exposure_data(
        exposure_data=sampled_exposures.tolist(),
        duration_in_seconds=3600,  # 1 hour
        num_paths_per_exposure=100,
        random_seed=42
    )
    
    exposures = results['exposures']
    probabilities = results['probabilities']
    drifts = results['mu_values']
    price_paths = results['price_paths']
    metrics = results['summary_stats']
    
    # Plot 1: Exposure-Response Mapping
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Risk Metrics: Exposure-Response Mapping', fontsize=16, fontweight='bold')
    
    # Exposure to Probability
    ax = axes[0, 0]
    scatter = ax.scatter(exposures, probabilities, alpha=0.7, c=exposures, cmap='RdYlBu_r', s=50)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Neutral (50%)')
    ax.axhline(y=0.5 + engine.k, color='orange', linestyle=':', alpha=0.8, label=f'Upper Bound ({0.5 + engine.k:.1%})')
    ax.axhline(y=0.5 - engine.k, color='orange', linestyle=':', alpha=0.8, label=f'Lower Bound ({0.5 - engine.k:.1%})')
    ax.set_xlabel('Net Exposure ($)', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Exposure → Probability Mapping', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Exposure Level')
    
    # Probability to Drift
    ax = axes[0, 1]
    scatter2 = ax.scatter(probabilities, drifts, alpha=0.7, c=np.abs(drifts), cmap='plasma', s=50)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Zero Drift')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Neutral Probability')
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_ylabel('Drift (μ)', fontsize=12)
    ax.set_title('Probability → Drift Mapping', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.colorbar(scatter2, ax=ax, label='|Drift| Magnitude')
    
    # Exposure to Mean Final Price
    ax = axes[1, 0]
    mean_final_prices = [m['mean_final_price'] for m in metrics]
    scatter3 = ax.scatter(exposures, mean_final_prices, alpha=0.7, c=mean_final_prices, cmap='viridis', s=50)
    ax.axhline(y=engine.S_0, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Starting Price ({engine.S_0:,.0f})')
    ax.set_xlabel('Net Exposure ($)', fontsize=12)
    ax.set_ylabel('Mean Final Price', fontsize=12)
    ax.set_title('Exposure → Expected Price Impact', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.colorbar(scatter3, ax=ax, label='Final Price')
    
    # Price Volatility vs Exposure
    ax = axes[1, 1]
    price_volatilities = [m['mean_realized_sigma'] for m in metrics]
    scatter4 = ax.scatter(exposures, price_volatilities, alpha=0.7, c=price_volatilities, cmap='Reds', s=50)
    ax.set_xlabel('Net Exposure ($)', fontsize=12)
    ax.set_ylabel('Price Volatility (Realized σ)', fontsize=12)
    ax.set_title('Exposure → Price Uncertainty', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=ax, label='Volatility')
    
    plt.tight_layout()
    plot1_path = f"{plots_dir}/1_exposure_response_mapping.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    print(f"Exposure-Response plot saved to {plot1_path}")
    plt.show()
    
    # Plot 2: Price Path Analysis
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Risk Metrics: Price Path Analysis', fontsize=16, fontweight='bold')
    
    # Sample Price Paths
    ax = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, min(8, len(exposures))))
    
    for i in range(min(8, len(exposures))):
        paths = price_paths[i]
        exposure = exposures[i]
        
        # Plot representative paths
        for j, path in enumerate(paths[:min(5, 30)]):
            alpha = 0.6 if j == 0 else 0.3
            linewidth = 2 if j == 0 else 1
            label = f'Exp: {exposure:,.0f}' if j == 0 else None
            ax.plot(path, color=colors[i], alpha=alpha, linewidth=linewidth, label=label)
    
    ax.axhline(y=engine.S_0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Starting Price')
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Index Price', fontsize=12)
    ax.set_title('Sample Price Paths by Exposure', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Final Price Distribution
    ax = axes[0, 1]
    all_final_prices = []
    for final_prices in price_paths:
        for path in final_prices:
            all_final_prices.append(path[-1])
    
    ax.hist(all_final_prices, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(x=engine.S_0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Starting Price')
    ax.set_xlabel('Final Price', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Final Price Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Risk Metrics
    ax = axes[1, 0]
    price_ranges = [max([path[-1] for path in paths]) - min([path[-1] for path in paths]) for paths in price_paths]
    scatter5 = ax.scatter(exposures, price_ranges, alpha=0.7, c=price_ranges, cmap='Oranges', s=50)
    ax.set_xlabel('Net Exposure ($)', fontsize=12)
    ax.set_ylabel('Price Range (Max - Min)', fontsize=12)
    ax.set_title('Exposure → Price Risk Range', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter5, ax=ax, label='Price Range')
    
    # Parameter Summary
    ax = axes[1, 1]
    ax.text(0.05, 0.95, 'Engine Configuration', fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.05, 0.85, f'Optimized Parameters:', fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.text(0.05, 0.78, f'• σ (volatility): {engine.sigma:.1%}', fontsize=10, transform=ax.transAxes)
    ax.text(0.05, 0.71, f'• Scale: {engine.scale:,.0f}', fontsize=10, transform=ax.transAxes)
    ax.text(0.05, 0.64, f'• k (range): ±{engine.k:.1%}', fontsize=10, transform=ax.transAxes)
    ax.text(0.05, 0.57, f'• Smoothness: {engine.smoothness_factor:.1f}', fontsize=10, transform=ax.transAxes)
    ax.text(0.05, 0.50, f'• S₀: {engine.S_0:,.0f}', fontsize=10, transform=ax.transAxes)
    
    ax.text(0.05, 0.38, f'Risk Controls:', fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.text(0.05, 0.31, f'• Noise Injection: {engine.noise_injection_level:.1%}', fontsize=10, transform=ax.transAxes)
    ax.text(0.05, 0.24, f'• Time Horizon: {engine.T * 365 * 24:.1f} hours', fontsize=10, transform=ax.transAxes)
    
    ax.text(0.05, 0.12, f'Performance:', fontsize=12, fontweight='bold', transform=ax.transAxes)
    validation_rate = np.mean([m['mu_validation_rate'] for m in metrics])
    ax.text(0.05, 0.05, f'• Validation Rate: {validation_rate:.1%}', fontsize=10, transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plot2_path = f"{plots_dir}/2_price_path_analysis.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    print(f"Price Path Analysis plot saved to {plot2_path}")
    plt.show()
    
    # Plot 3: Risk and Performance Metrics
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Risk Metrics: Performance & Risk Assessment', fontsize=16, fontweight='bold')
    
    # Responsiveness Analysis
    ax = axes[0, 0]
    correlations = []
    for i, (exp, prob) in enumerate(zip(exposures, probabilities)):
        if i > 0:
            exp_change = exp - exposures[i-1]
            prob_change = prob - probabilities[i-1]
            if exp_change != 0:
                local_resp = abs(prob_change / exp_change) * 1000000  # Scale for visibility
                correlations.append(local_resp)
            else:
                correlations.append(0)
        else:
            correlations.append(0)
    
    ax.plot(range(len(correlations)), correlations, 'b-', alpha=0.7, linewidth=2)
    ax.set_xlabel('Exposure Sequence', fontsize=12)
    ax.set_ylabel('Local Responsiveness', fontsize=12)
    ax.set_title('Responsiveness Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Risk Distribution Heatmap
    ax = axes[0, 1]
    prob_bins = np.linspace(min(probabilities), max(probabilities), 10)
    drift_bins = np.linspace(min(drifts), max(drifts), 10)
    risk_matrix = np.zeros((len(prob_bins)-1, len(drift_bins)-1))
    
    for prob, drift in zip(probabilities, drifts):
        prob_idx = min(np.digitize(prob, prob_bins) - 1, len(prob_bins) - 2)
        drift_idx = min(np.digitize(drift, drift_bins) - 1, len(drift_bins) - 2)
        risk_matrix[prob_idx, drift_idx] += 1
    
    im = ax.imshow(risk_matrix, cmap='YlOrRd', aspect='auto', origin='lower')
    ax.set_xlabel('Drift Bins', fontsize=12)
    ax.set_ylabel('Probability Bins', fontsize=12)
    ax.set_title('Risk Distribution Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Frequency')
    
    # Volatility Analysis
    ax = axes[1, 0]
    volatilities = [m['mean_realized_sigma'] for m in metrics]
    ax.hist(volatilities, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(x=engine.sigma, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Target σ: {engine.sigma:.1%}')
    ax.set_xlabel('Realized Volatility', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Volatility Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Performance Summary
    ax = axes[1, 1]
    total_exposures = len(exposures)
    avg_prob = np.mean(probabilities)
    prob_range = max(probabilities) - min(probabilities)
    avg_volatility = np.mean(volatilities)
    
    ax.text(0.05, 0.9, 'Performance Summary', fontsize=14, fontweight='bold', transform=ax.transAxes)
    ax.text(0.05, 0.8, f'Total Scenarios: {total_exposures}', fontsize=11, transform=ax.transAxes)
    ax.text(0.05, 0.73, f'Avg Probability: {avg_prob:.3f}', fontsize=11, transform=ax.transAxes)
    ax.text(0.05, 0.66, f'Probability Range: {prob_range:.3f}', fontsize=11, transform=ax.transAxes)
    ax.text(0.05, 0.59, f'Avg Volatility: {avg_volatility:.1%}', fontsize=11, transform=ax.transAxes)
    
    # Risk indicators
    high_risk_scenarios = sum(1 for exp in exposures if abs(exp) > np.std(exposures) * 2)
    ax.text(0.05, 0.45, 'Risk Indicators:', fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.text(0.05, 0.38, f'High Risk Scenarios: {high_risk_scenarios}', fontsize=11, transform=ax.transAxes)
    ax.text(0.05, 0.31, f'Risk Ratio: {high_risk_scenarios/total_exposures:.1%}', fontsize=11, transform=ax.transAxes)
    
    # Optimization status
    ax.text(0.05, 0.17, 'Optimization Status:', fontsize=12, fontweight='bold', transform=ax.transAxes)
    ax.text(0.05, 0.1, f'Parameters Optimized: ✓', fontsize=11, transform=ax.transAxes)
    ax.text(0.05, 0.03, f'Validation Rate: {validation_rate:.1%}', fontsize=11, transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plot3_path = f"{plots_dir}/3_risk_performance_metrics.png"
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"Risk Performance Metrics plot saved to {plot3_path}")
    plt.show()


def generate_dynamic_index_path(
    engine: SupplyDemandIndexEngine,
    exposure_series: pd.Series,
    optimal_params: Dict[str, float],
    num_simulations: int = 30,
    random_seed: int = 1505
) -> None:
    """Generate dynamic index paths using optimal parameters"""
    print("Generating dynamic index paths with optimal parameters...")
    
    # Apply optimal parameters
    engine.sigma = optimal_params['sigma']
    engine.scale = optimal_params['scale']
    engine.k = optimal_params['k']
    engine.smoothness_factor = optimal_params['smoothness_factor']
    
    print(f"Applied optimal parameters:")
    print(f"  sigma: {engine.sigma:.1%}")
    print(f"  scale: {engine.scale:,.0f}")
    print(f"  k: {engine.k:.1%}")
    print(f"  smoothness_factor: {engine.smoothness_factor:.1f}")
    
    # Use the engine's built-in dynamic path generation
    all_paths = []
    
    for i in range(num_simulations):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Generated simulation {i + 1}/{num_simulations}")

        try:
            # Generate path using the engine's method
            path, drift_path, smoothed_drift_path, probability_path = engine.generate_dynamic_exposure_path(
                exposure_series=exposure_series,
                random_seed=random_seed + i,
                ma_window=12
            )
            all_paths.append(path)
            
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

    # Create time array (5-minute intervals)
    time_points = np.arange(len(mean_path))

    # Create single plot visualization
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    
    # Plot individual paths
    print("  Plotting individual paths...")
    for i in range(min(15, len(all_paths))):
        ax.plot(time_points, all_paths[i], alpha=0.15, color="steelblue", linewidth=0.8)

    # Plot mean and bands
    print("  Plotting mean paths...")
    ax.plot(time_points, mean_path, color="darkblue", linewidth=2.5, label="Mean Index Path (Optimized)")
    ax.fill_between(
        time_points,
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

    ax.set_xlabel("Time (5-minute intervals)", fontsize=12)
    ax.set_ylabel("Index Price", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_title(
        f"Risk-Optimized Supply-Demand Index: Dynamic Response to Vol 75 Exposure\n"
        f"({len(all_paths)} simulations, {len(time_points)} data points)",
        fontsize=14,
    )

    # Create plots directory if it doesn't exist
    plots_dir = "plots/risk_metrics"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/4_dynamic_index_path.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Dynamic index path plot saved to {plots_dir}/4_dynamic_index_path.png")
    print(f"Generated {len(time_points)} index values")


def generate_parameter_classification_report() -> str:
    """
    Generate a report classifying parameters as fixed vs real-time adjustable
    
    Returns:
    --------
    str
        Parameter classification report
    """
    report = []
    report.append("=" * 80)
    report.append("PARAMETER CLASSIFICATION REPORT")
    report.append("Supply-Demand Index Engine")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Fixed Parameters (Should not be changed during operation)
    report.append("FIXED PARAMETERS (Set once, do not change during operation)")
    report.append("-" * 60)
    report.append("These parameters define the core engine behavior and should only be")
    report.append("changed during major system updates or reconfigurations.")
    report.append("")
    
    fixed_params = [
        ("S_0", 100_000, "Starting price reference point", "Critical for price calculations"),
        ("T", 1/(365*24), "Time horizon for calculations", "Affects all drift computations"),
        ("dt", 1/(86_400*365), "Time step size", "Core simulation parameter"),
    ]
    
    for param, value, description, reason in fixed_params:
        if isinstance(value, float):
            if param in ['T', 'dt']:
                report.append(f"• {param:<20}: {value:.2e} ({description})")
            else:
                report.append(f"• {param:<20}: {value:.3f} ({description})")
        else:
            report.append(f"• {param:<20}: {value} ({description})")
        report.append(f"  └─ Reason: {reason}")
        report.append("")
    
    # Real-time Adjustable Parameters
    report.append("REAL-TIME ADJUSTABLE PARAMETERS")
    report.append("-" * 60)
    report.append("These parameters can be adjusted during operation based on market")
    report.append("conditions, risk levels, or exploitation attempts.")
    report.append("")
    
    # High Priority (Adjust frequently)
    report.append("HIGH PRIORITY - Adjust Frequently (Every few hours)")
    report.append("~" * 50)
    
    high_priority = [
        ("noise_injection_level", 0.0, "Anti-exploitation noise", 
         "Increase if exploitation detected, decrease if too much randomness"),
        ("scale", 150_000, "Exposure sensitivity", 
         "Increase during high volatility, decrease during calm periods"),
        ("k", 0.4, "Probability range", 
         "Adjust based on market regime and required responsiveness")
    ]
    
    for param, value, description, guidance in high_priority:
        if isinstance(value, float):
            if param == 'scale':
                report.append(f"• {param:<25}: {value:,.0f} ({description})")
            else:
                report.append(f"• {param:<25}: {value:.1%} ({description})")
        else:
            report.append(f"• {param:<25}: {value} ({description})")
        report.append(f"  └─ Guidance: {guidance}")
        report.append("")
    
    # Medium Priority (Adjust daily/weekly)
    report.append("MEDIUM PRIORITY - Adjust Daily/Weekly")
    report.append("~" * 40)
    
    medium_priority = [
        ("sigma", 0.3, "Base volatility", 
         "Adjust based on market volatility regime"),
        ("smoothness_factor", 2.0, "Transition smoothness",
         "Increase for smoother transitions, decrease for more responsive behavior")
    ]
    
    for param, value, description, guidance in medium_priority:
        if isinstance(value, float):
            if param == 'sigma':
                report.append(f"• {param:<25}: {value:.1%} ({description})")
            else:
                report.append(f"• {param:<25}: {value:.1f} ({description})")
        else:
            report.append(f"• {param:<25}: {value} ({description})")
        report.append(f"  └─ Guidance: {guidance}")
        report.append("")
    
    # Adjustment Triggers
    report.append("ADJUSTMENT TRIGGERS")
    report.append("-" * 30)
    report.append("Conditions that should trigger parameter adjustments:")
    report.append("")
    
    triggers = [
        ("High Exploitation Risk", "Increase noise_injection_level"),
        ("Low Responsiveness", "Decrease scale, increase k, reduce smoothness_factor"),
        ("High Volatility Period", "Increase sigma, reduce scale sensitivity"),
        ("Trending Market", "Adjust k and smoothness_factor"),
        ("New Market Regime", "Comprehensive parameter review and adjustment")
    ]
    
    for trigger, action in triggers:
        report.append(f"• {trigger}:")
        report.append(f"  └─ Action: {action}")
        report.append("")
    
    # Monitoring Recommendations
    report.append("MONITORING RECOMMENDATIONS")
    report.append("-" * 35)
    report.append("Key metrics to monitor for parameter adjustment decisions:")
    report.append("")
    
    monitoring = [
        "• Responsiveness Score (target: > 0.6)",
        "• Validation Rate (target: > 0.7)",
        "• Price Volatility vs Target",
        "• Autocorrelation in returns (target: < 0.1)",
        "• Pattern detection attempts",
        "• Market regime changes"
    ]
    
    for item in monitoring:
        report.append(item)
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def generate_comprehensive_report(
    optimal_params: Dict[str, float],
    optimization_results: Dict[str, Any],
    exposure_data: pd.Series
) -> str:
    """
    Generate comprehensive risk metrics report
    
    Parameters:
    -----------
    optimal_params : Dict[str, float]
        Optimal parameters from optimization
    optimization_results : Dict[str, Any]
        Results from parameter optimization
    exposure_data : pd.Series
        Original exposure data
        
    Returns:
    --------
    str
        Formatted comprehensive report
    """
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE RISK METRICS ANALYSIS REPORT")
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
    
    # Optimization Results
    report.append("PARAMETER OPTIMIZATION RESULTS")
    report.append("-" * 40)
    report.append("Optimal parameters found:")
    for param, value in optimal_params.items():
        if isinstance(value, float):
            if param in ['sigma', 'k']:
                report.append(f"  {param}: {value:.1%}")
            elif param == 'scale':
                report.append(f"  {param}: {value:,.0f}")
            else:
                report.append(f"  {param}: {value:.2f}")
        else:
            report.append(f"  {param}: {value}")
    report.append("")
    
    # Performance Improvement
    baseline_val = optimization_results.get('baseline_validation_rate', 0)
    optimized_val = optimization_results.get('optimized_validation_rate', 0)
    improvement = optimization_results.get('validation_improvement', 0)
    
    report.append("PERFORMANCE IMPROVEMENT")
    report.append("-" * 40)
    report.append(f"Baseline validation rate: {baseline_val:.1%}")
    report.append(f"Optimized validation rate: {optimized_val:.1%}")
    report.append(f"Improvement: {improvement:+.1%}")
    report.append("")
    
    # Risk Assessment
    report.append("RISK ASSESSMENT")
    report.append("-" * 40)
    
    if improvement > 0.05:
        risk_level = "LOW"
        risk_color = "✓"
    elif improvement > 0:
        risk_level = "MEDIUM"
        risk_color = "~"
    else:
        risk_level = "HIGH"
        risk_color = "⚠"
    
    report.append(f"Overall improvement: {improvement:+.1%}")
    report.append(f"Risk level: {risk_color} {risk_level}")
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 40)
    
    if improvement > 0.05:
        report.append("✓ Significant optimization achieved - parameters are well-tuned")
    elif improvement > 0:
        report.append("~ Moderate improvement - consider further optimization")
    else:
        report.append("⚠ Limited improvement - review optimization approach")
    
    # Parameter-specific recommendations
    report.append("")
    report.append("PARAMETER RECOMMENDATIONS:")
    
    if optimal_params.get('sigma', 0) > 0.5:
        report.append("• Consider reducing volatility parameter for stability")
    
    if optimal_params.get('scale', 0) < 100000:
        report.append("• Scale parameter may be too sensitive - monitor for over-reaction")
    
    if optimal_params.get('k', 0) > 0.45:
        report.append("• High k value - ensure sufficient responsiveness")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """
    Main function to run the complete risk metrics analysis
    """
    print("Starting Risk Metrics Analysis...")
    print("=" * 60)
    
    try:
        # Step 1: Load trade data
        print("Step 1: Loading Vol 75 trade data...")
        trades_df = load_vol75_trade_data()
        
        # Step 2: Calculate net exposure
        print("\nStep 2: Calculating net exposure...")
        exposure_data = calculate_net_exposure(trades_df, window_minutes=5)
        
        # Step 3: Optimize parameters
        print("\nStep 3: Optimizing parameters...")
        optimal_params, optimization_results = optimize_parameters(
            exposure_data=exposure_data,
            sample_size=50,
            n_iterations=30
        )
        
        # Step 4: Create engine with optimal parameters
        print("\nStep 4: Creating optimized engine...")
        engine = SupplyDemandIndexEngine(
            sigma=optimal_params['sigma'],
            scale=optimal_params['scale'],
            k=optimal_params['k'],
            smoothness_factor=optimal_params['smoothness_factor']
        )
        
        # Step 5: Create all risk plots
        print("\nStep 5: Creating comprehensive risk plots...")
        create_all_risk_plots(
            engine=engine,
            exposure_data=exposure_data,
            optimal_params=optimal_params,
            sample_size=20
        )
        
        # Step 6: Generate dynamic index paths
        print("\nStep 6: Generating dynamic index paths...")
        generate_dynamic_index_path(
            engine=engine,
            exposure_series=exposure_data,
            optimal_params=optimal_params,
            num_simulations=30,
            random_seed=1505
        )
        
        # Step 7: Generate reports
        print("\nStep 7: Generating reports...")
        
        # Comprehensive report
        comprehensive_report = generate_comprehensive_report(
            optimal_params=optimal_params,
            optimization_results=optimization_results,
            exposure_data=exposure_data
        )
        print(comprehensive_report)
        
        # Parameter classification report
        classification_report = generate_parameter_classification_report()
        
        # Save reports
        with open('risk_metrics_comprehensive_report.txt', 'w') as f:
            f.write(comprehensive_report)
        
        with open('risk_metrics_parameter_classification.txt', 'w') as f:
            f.write(classification_report)
        
        # Save optimization results
        results_data = {
            'optimal_params': optimal_params,
            'optimization_results': optimization_results,
            'data_summary': {
                'total_trades': len(trades_df),
                'total_exposure_points': len(exposure_data),
                'exposure_range': [float(exposure_data.min()), float(exposure_data.max())],
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
        with open('risk_metrics_results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print("\nRisk Metrics Analysis complete! Results saved to:")
        print("  - risk_metrics_comprehensive_report.txt")
        print("  - risk_metrics_parameter_classification.txt")
        print("  - risk_metrics_results.json")
        print("  - plots/risk_metrics/ (visualization files)")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
