"""
Enhanced Supply-Demand Index Engine
Consolidated from adjusted and risk engines with noise injection capability

This module provides a clean, enhanced engine for supply-demand index generation with:
1. Smooth, non-linear exposure-to-probability mapping
2. Adaptive drift calculation with market psychology
3. Standard GBM price path generation (no memory/mean reversion)
4. Optional noise injection for anti-exploitation
5. All functions needed for risk testing

Author: Cline
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from typing import Tuple, List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class SupplyDemandIndexEngine:
    """
    Enhanced engine for generating supply-demand responsive index prices
    with optional noise injection and comprehensive testing capabilities
    """

    def __init__(
        self,
        sigma: float = 0.3,
        scale: float = 150_000,
        k: float = 0.4,
        smoothness_factor: float = 2.0,
        T: float = 1 / (365 * 24),  # 1 hour in years
        S_0: float = 100_000,
        dt: float = 1 / (86_400 * 365),  # 1 second in years
        noise_injection_level: float = 0.0,  # Optional noise injection
    ):
        """
        Initialize the Enhanced Supply-Demand Index Engine

        Parameters:
        -----------
        sigma : float
            Base volatility (annualized)
        scale : float
            Exposure sensitivity parameter for mapping
        k : float
            Maximum deviation from 0.5 in probability mapping
        smoothness_factor : float
            Controls smoothness of transitions in probability mapping
        T : float
            Time horizon for probability calculations (in years)
        S_0 : float
            Initial index price
        dt : float
            Time step size (in years)
        noise_injection_level : float
            Level of noise injection for anti-exploitation (0.0 = no noise)
        """
        self.sigma = sigma
        self.scale = scale
        self.k = k
        self.smoothness_factor = smoothness_factor
        self.T = T
        self.S_0 = S_0
        self.dt = dt
        self.noise_injection_level = noise_injection_level

        # Storage for results
        self.results = {}

    def add_noise_injection(self, base_value: float, noise_level: float = None) -> float:
        """
        Add controlled noise to prevent exploitation while maintaining responsiveness
        
        Parameters:
        -----------
        base_value : float
            Base value to add noise to
        noise_level : float, optional
            Noise level (defaults to self.noise_injection_level)
            
        Returns:
        --------
        float
            Value with anti-exploitation noise added
        """
        if noise_level is None:
            noise_level = self.noise_injection_level
            
        if noise_level <= 0:
            return base_value
            
        noise = np.random.normal(0, noise_level * abs(base_value))
        return base_value + noise

    def exposure_to_probability(self, exposure: float) -> float:
        """
        Map net exposure to probability using a smooth approach
        that combines multiple mathematical functions for natural transitions.

        This approach:
        1. Uses a combination of sigmoid and arctan functions for smoothness
        2. Incorporates adaptive sensitivity based on exposure magnitude
        3. Ensures continuous derivatives for realistic market behavior

        Parameters:
        -----------
        exposure : float
            Net exposure - higher exposure leads to higher probability of price increase

        Returns:
        --------
        float
            Probability of ending above current price (0-1)
        """
        # Normalize exposure with adaptive scaling
        normalized_exposure = exposure / self.scale

        # Apply sigmoid-based transformation with smoothness control
        base_sigmoid = 1 / (1 + np.exp(-self.smoothness_factor * normalized_exposure))

        # Apply arctan transformation for better behavior at extremes
        arctan_component = np.arctan(normalized_exposure) / np.pi + 0.5

        # Blend the two components with adaptive weighting
        blend_weight = np.exp(-0.5 * normalized_exposure**2)  # Gaussian weight
        blended_probability = (
            blend_weight * base_sigmoid + (1 - blend_weight) * arctan_component
        )

        # Apply non-linear transformation to enhance market psychology
        if exposure >= 0:
            # Positive exposure: gradual confidence building
            psychology_factor = np.tanh(normalized_exposure / 2)
        else:
            # Negative exposure: faster fear response
            psychology_factor = -np.tanh(-normalized_exposure / 1.5)

        # Incorporate market psychology into probability
        adjusted_probability = blended_probability + 0.1 * psychology_factor

        # Scale to desired range [0.5-k, 0.5+k]
        scaled_probability = 0.5 + (adjusted_probability - 0.5) * (2 * self.k)

        # Apply noise injection if enabled
        final_probability = self.add_noise_injection(scaled_probability)

        # Ensure probability stays within bounds
        return np.clip(final_probability, 0.5 - self.k, 0.5 + self.k)

    def compute_mu_from_probability(
        self, P: float, S: float = None, K: float = None
    ) -> float:
        """
        Compute drift parameter from desired probability using an approach
        that combines multiple distribution models for a more natural and smooth response.

        Parameters:
        -----------
        P : float
            Desired probability of ending above strike
        S : float, optional
            Current spot price (defaults to S_0)
        K : float, optional
            Strike price (defaults to S for "above current" probability)

        Returns:
        --------
        float
            Required drift parameter (mu) with scaling
        """
        if S is None:
            S = self.S_0
        if K is None:
            K = S  # For "above current price" probability

        # Handle edge cases with a smooth approach
        P = np.clip(P, 0.001, 0.999)

        # Calculate probability distance from neutral (0.5)
        prob_distance = P - 0.5
        abs_prob_distance = abs(prob_distance)

        # Determine market sentiment based on probability
        bullish_sentiment = prob_distance > 0

        # Create a blend of normal and t-distributions based on probability extremity
        extremity_factor = np.tanh(4 * abs_prob_distance)  # 0 to ~1

        # Calculate quantile using blended distribution
        if extremity_factor < 0.5:
            # More normal-like for moderate probabilities
            d2 = norm.ppf(P)
        else:
            # More t-distribution-like for extreme probabilities
            df = 5 - 3 * extremity_factor  # Degrees of freedom (2 to 5)
            d2 = t.ppf(P, df)

            # Adjust for t-distribution vs normal distribution
            t_adjustment = 1.0 + 0.2 * (1.0 - df / 5) * d2**2 / df
            d2 = d2 * t_adjustment

        # Calculate base drift using the Black-Scholes relationship
        base_mu = (
            d2 * self.sigma * np.sqrt(self.T) - np.log(S / K)
        ) / self.T + 0.5 * self.sigma**2

        # Apply scaling based on market psychology
        if bullish_sentiment:
            # Positive sentiment (bullish)
            psychology_factor = 1.0 + np.sin(np.pi / 2 * prob_distance) * 0.5
        else:
            # Negative sentiment (bearish)
            psychology_factor = 1.0 + np.sin(np.pi / 2 * abs_prob_distance) * 0.8

        # Apply smooth scaling
        scaled_mu = base_mu * psychology_factor

        # Add small non-linear component for more natural behavior
        non_linear_component = np.sign(scaled_mu) * (abs_prob_distance**1.5) * 10

        # Blend components with smooth transition
        final_mu = scaled_mu + non_linear_component * 0.2

        # Apply noise injection if enabled
        final_mu = self.add_noise_injection(final_mu)

        return final_mu

    def generate_price_path(
        self, mu: float, duration_in_seconds: int, random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate price path using standard Geometric Brownian Motion (GBM)
        with enhanced probability mapping and drift calculation.

        Parameters:
        -----------
        mu : float
            Drift parameter (annualized)
        duration_in_seconds : int
            Simulation duration in seconds
        random_seed : int, optional
            Random seed for reproducibility

        Returns:
        --------
        np.ndarray
            Price path array
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        num_second_per_tick = int(1 / self.dt / (86_400 * 365))
        num_step = int(duration_in_seconds / num_second_per_tick)

        # Generate log returns using standard GBM
        price_path = (mu - self.sigma**2 / 2) * self.dt + self.sigma * np.sqrt(
            self.dt
        ) * np.random.normal(0, 1, size=num_step)

        # Convert to price levels
        return self.S_0 * np.exp(np.cumsum(price_path))

    def generate_dynamic_exposure_path(
        self,
        exposure_series: pd.Series,
        random_seed: Optional[int] = None,
        ma_window: int = 12,  # Default 12-point moving average (1 hour with 5min data)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a price path that responds to time-varying exposure data
        with advanced smoothing and regime detection.

        Parameters:
        -----------
        exposure_series : pd.Series
            Time series of exposure values with datetime index
        random_seed : int, optional
            Random seed for reproducibility
        ma_window : int, optional
            Window size for moving average of drift parameter (default: 12)

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (price_path, drift_path, smoothed_drift_path, probability_path)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize arrays
        n_steps = len(exposure_series)
        price_path = np.zeros(n_steps + 1)
        drift_path = np.zeros(n_steps)
        smoothed_drift_path = np.zeros(n_steps)
        probability_path = np.zeros(n_steps)

        # Set initial price
        price_path[0] = self.S_0

        # First pass: calculate raw drift values directly from exposure
        for i, (timestamp, exposure) in enumerate(exposure_series.items()):
            # Map exposure to probability using approach
            probability = self.exposure_to_probability(exposure)
            probability_path[i] = probability

            # Map probability to drift using approach
            drift_path[i] = self.compute_mu_from_probability(probability)

        # Second pass: Apply weighted moving average to drift
        for i in range(n_steps):
            # Calculate moving average window bounds
            start_idx = max(0, i - ma_window + 1)
            # Get drift values in window
            window_values = drift_path[start_idx : i + 1]

            # Calculate weighted moving average with exponentially increasing weights
            window_size = len(window_values)
            if window_size > 1:
                # Create weights that increase exponentially (newer values have higher weights)
                alpha = 0.1  # Smoothing factor
                weights = np.array(
                    [(1 - alpha) ** j for j in range(window_size - 1, -1, -1)]
                )
                # Normalize weights to sum to 1
                weights = weights / np.sum(weights)
                # Apply weighted average
                smoothed_drift_path[i] = np.sum(window_values * weights)
            else:
                # If only one value, no weighting needed
                smoothed_drift_path[i] = window_values[0]

        # Third pass: generate price path using standard GBM
        for i in range(n_steps):
            current_price = price_path[i]
            mu = smoothed_drift_path[i]

            # Generate step using standard GBM (no memory or mean reversion)
            drift_term = (mu - 0.5 * self.sigma**2) * self.dt
            diffusion_term = self.sigma * np.sqrt(self.dt) * np.random.normal(0, 1)

            # Update price
            log_return = drift_term + diffusion_term
            price_path[i + 1] = current_price * np.exp(log_return)

        return price_path, drift_path, smoothed_drift_path, probability_path

    def validate_statistics(
        self,
        price_path: np.ndarray,
        expected_mu: float,
    ) -> Dict[str, Any]:
        """
        Validate that simulated path matches expected drift with metrics
        that better capture the behavior of the stochastic process.

        Parameters:
        -----------
        price_path : np.ndarray
            Simulated price path
        expected_mu : float
            Expected drift parameter

        Returns:
        --------
        dict
            Validation results and statistics
        """
        # Calculate log returns
        log_returns = np.diff(np.log(price_path))

        # Realized volatility
        realized_sigma = log_returns.std(ddof=1) / np.sqrt(self.dt)

        # Realized drift
        realized_drift = log_returns.mean() / self.dt
        realized_mu = realized_drift + 0.5 * realized_sigma**2

        # Calculate total return
        total_return = (price_path[-1] / price_path[0]) - 1

        # Enhanced validation approach:
        # 1. Check if drift direction is correct
        # 2. Check if magnitude is within reasonable bounds
        if expected_mu > 0:
            mu_valid_direction = price_path[-1] > price_path[0]
        elif expected_mu < 0:
            mu_valid_direction = price_path[-1] < price_path[0]
        else:  # expected_mu == 0
            mu_valid_direction = abs(total_return) < 0.01  # 1% threshold

        # Check if magnitude is reasonable (within 50% of expected)
        expected_return = np.exp(expected_mu * self.dt * len(log_returns)) - 1
        mu_valid_magnitude = abs(total_return - expected_return) < 0.5 * abs(
            expected_return
        )

        # Combined validation
        mu_valid = mu_valid_direction and mu_valid_magnitude

        # Calculate relative error
        mu_error = abs(realized_mu - expected_mu)

        return {
            "realized_sigma": realized_sigma,
            "realized_mu": realized_mu,
            "expected_mu": expected_mu,
            "mu_valid": mu_valid,
            "mu_valid_direction": mu_valid_direction,
            "mu_valid_magnitude": mu_valid_magnitude,
            "mu_error": mu_error,
            "realized_drift": realized_drift,
            "path_length": len(price_path),
            "final_price": price_path[-1],
            "total_return": total_return,
            "expected_return": expected_return,
        }

    def process_exposure_data(
        self,
        exposure_data: List[float],
        duration_in_seconds: int = 3600,
        num_paths_per_exposure: int = 100,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Process multiple exposure scenarios and generate corresponding price paths
        using the stochastic process.

        Parameters:
        -----------
        exposure_data : list
            List of net exposure values to process
        duration_in_seconds : int
            Duration for each price path simulation
        num_paths_per_exposure : int
            Number of Monte Carlo paths per exposure level
        random_seed : int
            Base random seed

        Returns:
        --------
        dict
            Complete results including paths, statistics, and validation
        """
        results = {
            "exposures": exposure_data,
            "probabilities": [],
            "mu_values": [],
            "price_paths": [],
            "mean_paths": [],
            "validation_results": [],
            "summary_stats": [],
        }

        for i, exposure in enumerate(exposure_data):
            # Step 1: Map exposure to probability
            prob = self.exposure_to_probability(exposure)
            results["probabilities"].append(prob)

            # Step 2: Convert probability to drift
            mu = self.compute_mu_from_probability(prob)
            results["mu_values"].append(mu)

            # Step 3: Generate multiple price paths
            paths = []
            validations = []

            for j in range(num_paths_per_exposure):
                path = self.generate_price_path(
                    mu=mu,
                    duration_in_seconds=duration_in_seconds,
                    random_seed=random_seed + i * 1000 + j,
                )
                paths.append(path)

                # Validate each path
                validation = self.validate_statistics(path, mu)
                validations.append(validation)

            results["price_paths"].append(paths)
            results["validation_results"].append(validations)

            # Calculate mean path
            mean_path = np.mean(paths, axis=0)
            results["mean_paths"].append(mean_path)

            # Summary statistics across all paths for this exposure
            summary = {
                "exposure": exposure,
                "probability": prob,
                "mu": mu,
                "mean_realized_sigma": np.mean(
                    [v["realized_sigma"] for v in validations]
                ),
                "mean_realized_mu": np.mean([v["realized_mu"] for v in validations]),
                "mean_final_price": np.mean([v["final_price"] for v in validations]),
                "mean_total_return": np.mean([v["total_return"] for v in validations]),
                "mu_validation_rate": np.mean([v["mu_valid"] for v in validations]),
            }
            results["summary_stats"].append(summary)

        self.results = results
        return results

    def calculate_risk_metrics(self, index_values: np.ndarray, exposures: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for the index
        
        Parameters:
        -----------
        index_values : np.ndarray
            Index price values
        exposures : np.ndarray
            Corresponding exposure values
            
        Returns:
        --------
        Dict[str, float]
            Risk metrics
        """
        # 1. Responsiveness Score
        if len(index_values) > 1 and len(exposures) == len(index_values) - 1:
            index_returns = np.diff(index_values) / index_values[:-1]
            correlation = np.corrcoef(exposures, index_returns)[0, 1]
            responsiveness = abs(correlation) if not np.isnan(correlation) else 0.0
        else:
            responsiveness = 0.0

        # 2. Exploitability Score (lower is better)
        if len(index_values) > 10:
            returns = np.diff(np.log(index_values))
            
            # Autocorrelation (should be low)
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            autocorr = abs(autocorr) if not np.isnan(autocorr) else 0.0
            
            # Pattern detection using entropy
            hist, _ = np.histogram(returns, bins=20)
            hist = hist / np.sum(hist)
            hist = hist[hist > 0]  # Remove zero probabilities
            entropy = -np.sum(hist * np.log(hist))
            max_entropy = np.log(len(hist))
            pattern_score = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
            
            exploitability = (autocorr + pattern_score) / 2
        else:
            exploitability = 0.5

        # 3. Risk Score (lower is better) - Focus on directional bias
        if len(index_values) > 1:
            returns = np.diff(index_values) / index_values[:-1]
            
            # Directional bias risk (trending in one direction)
            cumulative_return = np.sum(returns)
            directional_bias = abs(cumulative_return)
            
            # Maximum drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))
            
            # Trend persistence
            signs = np.sign(returns)
            sign_changes = np.sum(np.diff(signs) != 0)
            trend_persistence = 1.0 - (sign_changes / max(1, len(returns) - 1))
            
            # Skewness risk
            if len(returns) > 3:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    skewness = np.mean(((returns - mean_return) / std_return) ** 3)
                    skewness_risk = abs(skewness)
                else:
                    skewness_risk = 0.0
            else:
                skewness_risk = 0.0
            
            # Combined risk score
            risk_score = (
                directional_bias * 2.0 +
                max_drawdown * 1.5 +
                trend_persistence * 1.0 +
                skewness_risk * 0.5
            )
        else:
            risk_score = 0.0

        # 4. Unpredictability Score (higher is better)
        if len(index_values) > 10:
            returns = np.diff(np.log(index_values))
            
            # Entropy measure
            hist, _ = np.histogram(returns, bins=20)
            hist = hist / np.sum(hist)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log(hist))
            max_entropy = np.log(len(hist))
            entropy_score = entropy / max_entropy if max_entropy > 0 else 0.0
            
            # Randomness measure using runs test approximation
            median_return = np.median(returns)
            runs = np.sum(np.diff((returns > median_return).astype(int)) != 0) + 1
            n = len(returns)
            expected_runs = 2 * np.sum(returns > median_return) * np.sum(returns <= median_return) / n + 1
            randomness_score = min(1.0, runs / expected_runs) if expected_runs > 0 else 0.0
            
            unpredictability = (entropy_score + randomness_score) / 2
        else:
            unpredictability = 0.5

        return {
            'responsiveness': responsiveness,
            'exploitability': exploitability,
            'risk': risk_score,
            'unpredictability': unpredictability
        }

    def print_validation_summary(self, results: Optional[Dict] = None) -> None:
        """
        Print a summary of validation results

        Parameters:
        -----------
        results : dict, optional
            Results from process_exposure_data (uses self.results if None)
        """
        if results is None:
            results = self.results

        if not results:
            print("No results to summarize. Run process_exposure_data first.")
            return

        print("=" * 60)
        print("ENHANCED SUPPLY-DEMAND INDEX VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Engine Parameters:")
        print(f"  σ (volatility): {self.sigma:.1%}")
        print(f"  Scale: {self.scale:,.0f}")
        print(f"  k (max deviation): {self.k:.2f}")
        print(f"  T (time horizon): {self.T * 365 * 24:.1f} hours")
        print(f"  S₀ (initial price): {self.S_0:,.0f}")
        print(f"  Smoothness Factor: {self.smoothness_factor:.1f}")
        print(f"  Noise Injection: {self.noise_injection_level:.1%}")
        print()

        summary_stats = results["summary_stats"]

        print(f"Processed {len(summary_stats)} exposure scenarios:")
        print("-" * 60)

        for i, stats in enumerate(summary_stats):
            print(f"Scenario {i + 1}:")
            print(f"  Exposure: {stats['exposure']:>10,.0f}")
            print(f"  Probability: {stats['probability']:>8.3f}")
            print(f"  μ (drift): {stats['mu']:>10.4f}")
            print(f"  μ validation: {stats['mu_validation_rate']:>6.1%}")
            print(f"  Mean return: {stats['mean_total_return']:>8.2%}")
            print()

        # Overall validation rate
        overall_mu_rate = np.mean([s["mu_validation_rate"] for s in summary_stats])

        print("-" * 60)
        print(f"Overall Validation Rate:")
        print(f"  μ (drift): {overall_mu_rate:.1%}")
        print("=" * 60)

    def generate_summary_report(self, results: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate a summary report of all results

        Parameters:
        -----------
        results : dict, optional
            Results from process_exposure_data (uses self.results if None)

        Returns:
        --------
        pd.DataFrame
            Summary report
        """
        if results is None:
            results = self.results

        if not results:
            raise ValueError("No results to report. Run process_exposure_data first.")

        df = pd.DataFrame(results["summary_stats"])

        # Add additional calculated columns
        df["mu_percentage"] = df["mu"] * 100
        df["sigma_error"] = abs(df["mean_realized_sigma"] - self.sigma)
        df["mu_error"] = abs(df["mean_realized_mu"] - df["mu"])
        df["return_percentage"] = df["mean_total_return"] * 100

        # Format for display
        df_display = df.copy()
        for col in [
            "probability",
            "mean_realized_sigma",
            "mu_validation_rate",
        ]:
            df_display[col] = df_display[col].round(3)
        for col in ["mu_percentage", "return_percentage"]:
            df_display[col] = df_display[col].round(2)
        for col in ["exposure", "mean_final_price"]:
            df_display[col] = df_display[col].round(1)

        return df_display


def run_example_analysis():
    """
    Example usage of the Enhanced Supply-Demand Index Engine
    """
    print("Running Enhanced Supply-Demand Index Example Analysis...")

    # Initialize engine with default parameters
    engine = SupplyDemandIndexEngine(
        sigma=0.3,  # 30% volatility
        scale=150_000,  # Exposure sensitivity
        k=0.4,  # Max probability deviation
        smoothness_factor=2.0,  # Controls smoothness of transitions
        noise_injection_level=0.015,  # 1.5% noise injection
    )

    # Define exposure scenarios
    exposure_scenarios = [-50_000, -25_000, 0, 25_000, 50_000]

    print(f"Processing {len(exposure_scenarios)} exposure scenarios...")
    print(f"Exposure scenarios: {exposure_scenarios}")

    # Process the data
    results = engine.process_exposure_data(
        exposure_data=exposure_scenarios,
        duration_in_seconds=3600,  # 1 hour
        num_paths_per_exposure=200,
        random_seed=42,
    )

    # Print validation summary
    engine.print_validation_summary()

    # Generate summary report
    report = engine.generate_summary_report()
    print("\nDetailed Summary Report:")
    print(report.to_string(index=False))

    print("Analysis complete!")
    return engine, results


if __name__ == "__main__":
    # Run example analysis
    engine, results = run_example_analysis()
