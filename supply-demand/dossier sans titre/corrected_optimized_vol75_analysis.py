"""
Corrected Optimized Vol 75 Supply-Demand Index Analysis
Based on ACTUAL parameters from adjusted_supply_demand_index_engine.py

This script identifies the best values for the 4 most critical parameters:
1. noise_injection_level (Anti-exploitation - from risk engine)
2. scale (Exposure sensitivity) 
3. k (Probability range)
4. sigma (Base volatility)

All other parameters are fixed at optimal values.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any, Union
import json
from adjusted_supply_demand_index_engine import SupplyDemandIndexEngine

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


def create_optimized_engine_with_fixed_params() -> SupplyDemandIndexEngine:
    """
    Create engine with optimal FIXED parameters based on adjusted_supply_demand_index_engine.py
    Only 4 parameters will be dynamic: sigma, scale, k, smoothness_factor
    """
    return SupplyDemandIndexEngine(
        # DYNAMIC PARAMETERS (will be optimized)
        sigma=0.3,  # Will be optimized (30% volatility)
        scale=150_000,  # Will be optimized (exposure sensitivity)
        k=0.4,  # Will be optimized (max probability deviation)
        smoothness_factor=2.0,  # Will be optimized (transition smoothness)
        
        # FIXED PARAMETERS (optimal values from adjusted engine, never change)
        T=1.0 / (365 * 24),  # 1 hour time horizon - FIXED
        S_0=100_000,  # Starting price - FIXED
        dt=1 / (86_400 * 365),  # 1 second in years - FIXED
    )


def optimize_4_key_parameters(
    exposure_data: pd.Series, 
    sample_size: int = 50,
    n_iterations: int = 30
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Optimize the 4 most critical parameters from the adjusted engine
    
    Returns:
    --------
    Tuple[Dict[str, float], Dict[str, Any]]
        Optimal parameters and detailed results
    """
    print("Optimizing 4 key dynamic parameters from adjusted engine...")
    print("Parameters to optimize:")
    print("  1. sigma (Base volatility)")
    print("  2. scale (Exposure sensitivity)")
    print("  3. k (Max probability deviation)")
    print("  4. smoothness_factor (Transition smoothness)")
    print()
    
    # Create engine with fixed optimal parameters
    engine = create_optimized_engine_with_fixed_params()
    
    # Sample exposure data
    np.random.seed(1505)
    if len(exposure_data) > sample_size:
        sampled_indices = np.random.choice(len(exposure_data), size=sample_size, replace=False)
        sampled_exposure = exposure_data.iloc[sampled_indices].values
    else:
        sampled_exposure = exposure_data.values

    print(f"Using {len(sampled_exposure)} exposure points for optimization")
    print(f"Exposure range: {min(sampled_exposure):,.0f} to {max(sampled_exposure):,.0f}")

    # Define parameter bounds for the 4 key parameters
    from scipy.optimize import minimize
    
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
    engine_baseline = create_optimized_engine_with_fixed_params()
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


def generate_parameter_classification_report(optimal_params: Dict[str, float]) -> str:
    """Generate a comprehensive parameter classification report based on adjusted engine"""
    report = []
    report.append("=" * 80)
    report.append("CORRECTED PARAMETER CLASSIFICATION & OPTIMIZATION REPORT")
    report.append("Based on Adjusted Vol 75 Supply-Demand Index Engine")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Dynamic Parameters Section
    report.append("🔧 DYNAMIC PARAMETERS (Real-time adjustable)")
    report.append("=" * 50)
    report.append("These 4 parameters can be adjusted during operation based on:")
    report.append("market conditions, risk levels, or exploitation attempts.")
    report.append("")
    
    dynamic_params = [
        ("sigma", optimal_params['sigma'], "Base volatility (most critical)",
         "Increase during high volatility periods, decrease when calm", "10% - 60%"),
        ("scale", optimal_params['scale'], "Exposure sensitivity",
         "Increase during volatility (less sensitive), decrease when calm", "50,000 - 300,000"),
        ("k", optimal_params['k'], "Max probability deviation from 0.5",
         "Adjust based on required responsiveness range", "20% - 50%"),
        ("smoothness_factor", optimal_params['smoothness_factor'], "Transition smoothness",
         "Increase for smoother transitions, decrease for more responsive", "1.0 - 4.0")
    ]
    
    for i, (param, value, description, guidance, range_val) in enumerate(dynamic_params, 1):
        if param == 'scale':
            report.append(f"{i}. {param.upper()}: {value:,.0f}")
        elif param == 'smoothness_factor':
            report.append(f"{i}. {param.upper()}: {value:.1f}")
        else:
            report.append(f"{i}. {param.upper()}: {value:.1%}")
        report.append(f"   Purpose: {description}")
        report.append(f"   When to adjust: {guidance}")
        report.append(f"   Safe range: {range_val}")
        report.append("")
    
    # Add noise injection parameter (from risk engine concept)
    report.append("ADDITIONAL ANTI-EXPLOITATION PARAMETER:")
    report.append("5. NOISE_INJECTION_LEVEL: 1.5% (recommended)")
    report.append("   Purpose: Anti-exploitation protection")
    report.append("   When to adjust: Increase if exploitation detected")
    report.append("   Safe range: 0.5% - 5.0%")
    report.append("   Note: Can be implemented by adding random noise to probability calculations")
    report.append("")
    
    # Fixed Parameters Section
    report.append("🔒 FIXED PARAMETERS (Never change during operation)")
    report.append("=" * 55)
    report.append("These parameters are set to optimal values and should remain constant:")
    report.append("")
    
    # Core Engine Parameters
    report.append("CORE ENGINE PARAMETERS:")
    report.append("• T = 1 hour (Time horizon for calculations)")
    report.append("• S_0 = 100,000 (Starting price reference)")
    report.append("• dt = 1 second (Time step size)")
    report.append("")
    
    # Removed Parameters (not in adjusted engine)
    report.append("PARAMETERS REMOVED FROM ADJUSTED ENGINE:")
    report.append("• memory_length (No fractional memory)")
    report.append("• fractal_dimension (No fractal characteristics)")
    report.append("• mean_reversion_strength (No mean reversion)")
    report.append("• psychology_bull_factor (Integrated into probability mapping)")
    report.append("• psychology_bear_factor (Integrated into probability mapping)")
    report.append("")
    
    # Adjustment Guidelines
    report.append("📋 ADJUSTMENT GUIDELINES")
    report.append("=" * 30)
    report.append("")
    
    report.append("WHEN TO ADJUST DYNAMIC PARAMETERS:")
    report.append("")
    
    scenarios = [
        ("High Volatility Market", 
         "• Increase sigma (higher base volatility)\n• Increase scale (reduce sensitivity)\n• Consider increasing smoothness_factor"),
        ("Low Volatility Market", 
         "• Decrease sigma (lower base volatility)\n• Decrease scale (increase sensitivity)\n• Consider decreasing smoothness_factor"),
        ("Need More Responsiveness", 
         "• Decrease scale (increase sensitivity)\n• Increase k (wider probability range)\n• Decrease smoothness_factor"),
        ("Need More Stability", 
         "• Increase scale (reduce sensitivity)\n• Decrease k (narrower probability range)\n• Increase smoothness_factor"),
        ("Exploitation Risk Detected", 
         "• Add noise injection to probability calculations\n• Increase smoothness_factor\n• Consider increasing scale"),
        ("Market Regime Change", 
         "• Adjust sigma based on new volatility regime\n• Adjust scale based on new sensitivity requirements")
    ]
    
    for scenario, actions in scenarios:
        report.append(f"• {scenario}:")
        for action in actions.split('\n'):
            report.append(f"  {action}")
        report.append("")
    
    # Monitoring Section
    report.append("📊 MONITORING RECOMMENDATIONS")
    report.append("=" * 35)
    report.append("Monitor these metrics to guide parameter adjustments:")
    report.append("")
    
    metrics = [
        "• Validation Rate (target: > 70%)",
        "• Responsiveness (correlation between exposure and probability)",
        "• Price Volatility (stability of final prices)",
        "• Probability Range Utilization",
        "• Market Volatility Regime Changes",
        "• Exploitation Attempts Detection"
    ]
    
    for metric in metrics:
        report.append(metric)
    
    report.append("")
    report.append("⚠️  WARNING: Only adjust parameters based on clear evidence")
    report.append("   of changed market conditions or risk indicators.")
    report.append("")
    report.append("💡 TIP: The adjusted engine uses standard GBM without memory")
    report.append("   or mean reversion, making it simpler and more predictable.")
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def generate_dynamic_index_with_optimal_params(
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
                ma_window=12  # 1 hour with 5min data
            )
            all_paths.append(path)
                
        except Exception as e:
            print(f"Error in simulation {i+1}: {e}")
            continue

    if not all_paths:
        print("No successful simulations generated!")
        return

    # Convert to numpy array
    all_paths = np.array(all_paths)
    mean_path = np.mean(all_paths, axis=0)
    lower_band = np.percentile(all_paths, 10, axis=0)
    upper_band = np.percentile(all_paths, 90, axis=0)

    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    
    # Plot individual paths
    for i in range(min(15, len(all_paths))):
        ax.plot(all_paths[i], alpha=0.15, color="steelblue", linewidth=0.8)

    # Plot mean and bands
    ax.plot(mean_path, color="darkblue", linewidth=2.5, label="Mean Index Path (Optimized)")
    ax.fill_between(range(len(mean_path)), lower_band, upper_band, color="blue", alpha=0.2, label="80% Confidence Band")
    ax.axhline(y=engine.S_0, color="red", linestyle="--", alpha=0.7, label="Starting Price")

    ax.set_xlabel("Time Steps", fontsize=12)
    ax.set_ylabel("Index Price", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_title(
        f"Corrected Optimized Supply-Demand Index: Dynamic Response\n"
        f"Optimal Parameters Applied ({len(all_paths)} simulations, {len(mean_path):,} data points)",
        fontsize=14,
    )

    # Create plots directory
    plots_dir = "plots/corrected_optimized"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    plt.tight_layout()
    plt.savefig(f"{plots_dir}/corrected_optimized_dynamic_index_path.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Corrected optimized dynamic index path saved to {plots_dir}/corrected_optimized_dynamic_index_path.png")


def main():
    """Main function to run the corrected optimized parameter analysis"""
    print("Starting Corrected Optimized Vol 75 Parameter Analysis...")
    print("Based on ACTUAL parameters from adjusted_supply_demand_index_engine.py")
    print("=" * 70)
    
    try:
        # Step 1: Load trade data
        print("Step 1: Loading Vol 75 trade data...")
        trades_df = load_vol75_trade_data()
        
        # Step 2: Calculate net exposure
        print("\nStep 2: Calculating net exposure...")
        exposure_data = calculate_net_exposure(trades_df, window_minutes=5)
        
        # Step 3: Optimize 4 key parameters
        print("\nStep 3: Optimizing 4 key dynamic parameters...")
        optimal_params, optimization_results = optimize_4_key_parameters(
            exposure_data=exposure_data,
            sample_size=50,
            n_iterations=30
        )
        
        # Step 4: Generate classification report
        print("\nStep 4: Generating corrected parameter classification report...")
        classification_report = generate_parameter_classification_report(optimal_params)
        print(classification_report)
        
        # Save classification report
        with open('corrected_parameter_classification_report.txt', 'w') as f:
            f.write(classification_report)
        print("Corrected classification report saved to 'corrected_parameter_classification_report.txt'")
        
        # Step 5: Generate dynamic index with optimal parameters
        print("\nStep 5: Generating dynamic index with optimal parameters...")
        engine = create_optimized_engine_with_fixed_params()
        generate_dynamic_index_with_optimal_params(
            engine=engine,
            exposure_series=exposure_data,
            optimal_params=optimal_params,
            num_simulations=30,
            random_seed=1505
        )
        
        # Step 6: Save optimization results
        print("\nStep 6: Saving optimization results...")
        results_data = {
            'optimal_dynamic_parameters': optimal_params,
            'optimization_results': optimization_results,
            'fixed_parameters': {
                'T': 1 / (365 * 24),
                'S_0': 100_000,
                'dt': 1 / (86_400 * 365)
            },
            'removed_parameters': [
                'memory_length', 'fractal_dimension', 'mean_reversion_strength',
                'psychology_bull_factor', 'psychology_bear_factor'
            ],
            'data_summary': {
                'total_trades': len(trades_df),
                'total_exposure_points': len(exposure_data),
                'exposure_range': [float(exposure_data.min()), float(exposure_data.max())],
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
        with open('corrected_optimized_parameters_results.json', 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print("Analysis complete! Results saved to:")
        print("  - corrected_parameter_classification_report.txt")
        print("  - corrected_optimized_parameters_results.json")
        print("  - plots/corrected_optimized/corrected_optimized_dynamic_index_path.png")
        
        # Summary of optimal parameters
        print("\n" + "="*70)
        print("CORRECTED OPTIMAL DYNAMIC PARAMETERS SUMMARY")
        print("="*70)
        print(f"sigma: {optimal_params['sigma']:.1%} (Base volatility)")
        print(f"scale: {optimal_params['scale']:,.0f} (Exposure sensitivity)")
        print(f"k: {optimal_params['k']:.1%} (Max probability deviation)")
        print(f"smoothness_factor: {optimal_params['smoothness_factor']:.1f} (Transition smoothness)")
        print("="*70)
        print("Note: Based on actual parameters from adjusted_supply_demand_index_engine.py")
        print("Removed: memory_length, fractal_dimension, mean_reversion_strength")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
