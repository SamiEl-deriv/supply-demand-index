import numpy as np
import pandas as pd
from analysis import IndexAnalysis

if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=252, freq='D')
    # Filter out weekends
    dates = dates[dates.dayofweek < 5]
    # Generate returns data with proper length matching filtered dates
    n_days = len(dates)
    simulated_returns = pd.Series(np.random.normal(0.001, 0.02, n_days), index=dates)
    benchmark_returns = pd.Series(np.random.normal(0.001, 0.015, n_days), index=dates)
    simulated_prices = 100 * (1 + simulated_returns).cumprod()

    analysis = IndexAnalysis()
    
    # Calculate static return distribution moments
    moments = analysis.calculate_return_moments(simulated_returns)
    print("Static Return Distribution Analysis:")
    print(f"Mean (annualized): {moments['mean']:.4f}")
    print(f"Volatility (annualized): {moments['volatility']:.4f}")
    print(f"Skewness: {moments['skewness']:.4f}")
    print(f"Kurtosis: {moments['kurtosis']:.4f}")
    print(f"Excess Kurtosis: {moments['excess_kurtosis']:.4f}")
    print(f"Jarque-Bera p-value: {moments['jarque_bera_pvalue']:.4f}")
    print(f"Is Normal: {moments['is_normal']}\n")

    # Calculate rolling moments with convergence analysis
    print("\nCalculating rolling moments with convergence analysis (window=126 days)...")
    rolling_moments = analysis.calculate_return_moments(simulated_returns, window=126, plot=True)
    
    # Print convergence analysis
    convergence = rolling_moments['convergence']
    print("\nMoment Convergence Analysis:")
    print(f"Overall Convergence: {convergence['overall_convergence']['is_converged']}")
    print(f"Convergence Threshold: {convergence['overall_convergence']['convergence_threshold']}")
    if not convergence['overall_convergence']['is_converged']:
        print("Non-converged moments:", convergence['overall_convergence']['non_converged_moments'])
    
    # Print detailed convergence metrics for each moment
    print("\nDetailed Convergence Metrics:")
    for name, metrics in convergence.items():
        if name != 'overall_convergence':
            print(f"\n{name}:")
            print(f"  Converged: {metrics['is_converged']}")
            print(f"  Rate of Change: {metrics['rate_of_change']:.6f}")
            print(f"  Stability: {metrics['stability']:.6f}")
    
    # Calculate tracking metrics
    print("Tracking Error:", analysis.calculate_tracking_error(simulated_returns, benchmark_returns))
    print("Information Ratio:", analysis.calculate_information_ratio(simulated_returns, benchmark_returns))
    
    # Calculate volatility and momentum
    print("\nRealized Volatility (latest):", analysis.calculate_realized_volatility(simulated_returns).iloc[-1])
    momentum_scores = analysis.calculate_momentum_score(simulated_returns)
    print("\nMomentum Scores:")
    for period, score in momentum_scores.items():
        print(f"{period}: {score:.4f}")
    
    # Analyze trends
    trend_metrics = analysis.analyze_trend(simulated_returns)
    print("\nTrend Analysis:")
    for metric, value in trend_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Calculate and plot autocorrelation
    print("\nAutocorrelation Analysis:")
    autocorr_results = analysis.calculate_autocorrelation(simulated_returns, lags=20)
    print("Significant ACF lags:", autocorr_results['significant_lags']['acf'])
    print("Significant PACF lags:", autocorr_results['significant_lags']['pacf'])
    
    # Plot drawdown
    analysis.visualize_drawdown(simulated_returns)
    
    # Plot rolling metrics
    rolling_beta = analysis.calculate_rolling_beta(simulated_returns, benchmark_returns)
    realized_vol = analysis.calculate_realized_volatility(simulated_returns)
    analysis.visualize_rolling_metrics(dates, rolling_beta, realized_vol)
