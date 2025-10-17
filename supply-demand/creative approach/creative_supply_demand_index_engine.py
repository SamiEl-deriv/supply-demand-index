"""
Supply-Demand Index Engine

This module implements an advanced approach to supply-demand index generation with:
1. Smooth, non-linear exposure-to-probability mapping using advanced mathematical functions
2. Adaptive drift calculation with market regime awareness
3. Sophisticated stochastic process with mean reversion and fractal characteristics
4. Comprehensive statistical validation and visualization

Author: Cline
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm, t, skewnorm
from typing import Tuple, List, Optional, Dict, Any, Union
import warnings

warnings.filterwarnings("ignore")


class SupplyDemandIndexEngine:
    """
    Engine for generating supply-demand responsive index prices
    with advanced mathematical models and smooth transitions
    """

    def __init__(
        self,
        sigma: float = 0.1,
        scale: float = 10_000,
        k: float = 0.45,
        T: float = 1 / (365 * 24),  # 1 hour in years
        S_0: float = 100_000,
        dt: float = 1 / (86_400 * 365),  # 1 second in years
        mean_reversion_strength: float = 0.05,  # Strength of mean reversion
        fractal_dimension: float = 1.5,  # Fractal dimension (1.5 = standard Brownian)
        memory_length: int = 20,  # Length of memory for fractional process
        regime_threshold: float = 0.3,  # Threshold for regime switching
        smoothness_factor: float = 2.0,  # Controls smoothness of transitions
    ):
        """
        Initialize the Supply-Demand Index Engine

        Parameters:
        -----------
        sigma : float
            Base volatility (annualized)
        scale : float
            Exposure sensitivity parameter for mapping
        k : float
            Maximum deviation from 0.5 in probability mapping
        T : float
            Time horizon for probability calculations (in years)
        S_0 : float
            Initial index price
        dt : float
            Time step size (in years)
        mean_reversion_strength : float
            Strength of mean reversion component (0 = no mean reversion)
        fractal_dimension : float
            Controls the roughness of the price path (1.5 = standard Brownian)
        memory_length : int
            Number of past steps to consider for fractional process
        regime_threshold : float
            Threshold for switching between normal and stressed regimes
        smoothness_factor : float
            Controls smoothness of transitions in probability mapping
        """
        self.sigma = sigma
        self.scale = scale
        self.k = k
        self.T = T
        self.S_0 = S_0
        self.dt = dt
        self.mean_reversion_strength = mean_reversion_strength
        self.fractal_dimension = fractal_dimension
        self.memory_length = memory_length
        self.regime_threshold = regime_threshold
        self.smoothness_factor = smoothness_factor
        self.hurst = 2 - fractal_dimension  # Hurst exponent from fractal dimension

        # Storage for results
        self.results = {}

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
        # Small exposures get more attention, large exposures saturate gradually
        normalized_exposure = exposure / self.scale

        # Apply sigmoid-based transformation with smoothness control
        # This creates a smooth S-curve with controlled steepness
        base_sigmoid = 1 / (1 + np.exp(-self.smoothness_factor * normalized_exposure))

        # Apply arctan transformation for better behavior at extremes
        # This ensures smoother tails than pure sigmoid
        arctan_component = np.arctan(normalized_exposure) / np.pi + 0.5

        # Blend the two components with adaptive weighting
        # This creates a natural transition between different regimes
        blend_weight = np.exp(-0.5 * normalized_exposure**2)  # Gaussian weight
        blended_probability = (
            blend_weight * base_sigmoid + (1 - blend_weight) * arctan_component
        )

        # Apply non-linear transformation to enhance market psychology
        # Markets often react differently to positive vs negative exposures
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

        # Ensure probability stays within bounds
        return np.clip(scaled_probability, 0.5 - self.k, 0.5 + self.k)

    def compute_mu_from_probability(
        self, P: float, S: float = None, K: float = None
    ) -> float:
        """
        Compute drift parameter from desired probability using an approach
        that combines multiple financial theories for a more natural and smooth response.

        This implementation:
        1. Blends multiple distribution models for realistic market behavior
        2. Incorporates market psychology factors
        3. Uses adaptive scaling based on probability distance from neutral

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
        # This creates fat tails when probabilities are extreme
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
        # but with a twist for more natural market behavior
        base_mu = (
            d2 * self.sigma * np.sqrt(self.T) - np.log(S / K)
        ) / self.T + 0.5 * self.sigma**2

        # Apply scaling based on market psychology
        # This creates a more natural response curve
        if bullish_sentiment:
            # Positive sentiment (bullish)
            # Markets tend to rise gradually (climb the wall of worry)
            psychology_factor = 1.0 + np.sin(np.pi / 2 * prob_distance) * 0.5
        else:
            # Negative sentiment (bearish)
            # Markets tend to fall sharply (take the elevator down)
            psychology_factor = 1.0 + np.sin(np.pi / 2 * abs_prob_distance) * 0.8

        # Apply smooth scaling
        scaled_mu = base_mu * psychology_factor

        # Add small non-linear component for more natural behavior
        # This creates subtle variations that make the model more realistic
        non_linear_component = np.sign(scaled_mu) * (abs_prob_distance**1.5) * 10

        # Blend components with smooth transition
        final_mu = scaled_mu + non_linear_component * 0.2

        return final_mu

    def generate_price_path(
        self, mu: float, duration_in_seconds: int, random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate price path using a stochastic process that combines:
        1. Fractional Brownian motion for realistic market roughness
        2. Mean reversion with adaptive strength
        3. Regime-switching volatility with smooth transitions
        4. Long-memory effects for realistic autocorrelation

        Parameters:
        -----------
        mu : float
            Base drift parameter (annualized)
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

        # Calculate number of steps
        num_second_per_tick = int(1 / self.dt / (86_400 * 365))
        num_step = int(duration_in_seconds / num_second_per_tick)

        # Initialize price path array
        price_path = np.zeros(num_step + 1)
        price_path[0] = self.S_0

        # Initialize arrays for fractional process
        increments = np.zeros(num_step)
        regime_state = np.zeros(num_step)  # 0 to 1 continuous regime state

        # Generate correlated random increments for fractional Brownian motion
        # This creates realistic market roughness with long-memory effects
        for i in range(num_step):
            # Generate standard normal random number
            z = np.random.normal(0, 1)

            # For first few steps, use standard Brownian motion
            if i < self.memory_length:
                increments[i] = z
            else:
                # Apply fractional Brownian motion with memory
                # This creates realistic autocorrelation in returns
                memory_weights = np.power(
                    np.arange(1, self.memory_length + 1, 1), self.hurst - 1.5
                )
                memory_weights = memory_weights / np.sum(memory_weights)

                # Blend new innovation with memory of past innovations
                memory_effect = np.sum(
                    memory_weights * increments[i - self.memory_length : i]
                )
                increments[i] = 0.7 * z + 0.3 * memory_effect

        # Generate path step by step with regime awareness
        for i in range(num_step):
            current_price = price_path[i]

            # Update regime state with smooth transitions
            if i > 0:
                # Calculate local volatility estimate
                if i > 10:
                    local_returns = np.diff(np.log(price_path[max(0, i - 10) : i + 1]))
                    local_vol = np.std(local_returns) / np.sqrt(self.dt)

                    # Target regime state based on volatility
                    target_regime = np.tanh((local_vol / self.sigma - 1) * 2)

                    # Smooth transition to target regime
                    regime_state[i] = 0.9 * regime_state[i - 1] + 0.1 * target_regime
                else:
                    regime_state[i] = regime_state[i - 1]  # Maintain previous state

            # Calculate adaptive volatility based on regime
            effective_sigma = self.sigma * (1 + 0.5 * regime_state[i])

            # Calculate mean reversion component with adaptive strength
            # Stronger mean reversion when far from initial price
            log_price_ratio = np.log(current_price / self.S_0)
            base_reversion = -self.mean_reversion_strength * log_price_ratio

            # Adaptive mean reversion - stronger when further from equilibrium
            adaptive_factor = 1.0 + 0.5 * np.tanh(abs(log_price_ratio) - 0.1)
            mean_reversion = base_reversion * adaptive_factor

            # Combine drift components with smooth blending
            effective_drift = mu + mean_reversion

            # Generate step increment with fractional characteristics
            drift_term = (effective_drift - 0.5 * effective_sigma**2) * self.dt
            diffusion_term = effective_sigma * np.sqrt(self.dt) * increments[i]

            # Update price with log-normal step
            log_return = drift_term + diffusion_term
            price_path[i + 1] = current_price * np.exp(log_return)

        return price_path

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

        # Second pass: apply advanced smoothing to drift
        # Using wavelet-inspired multi-scale smoothing
        for i in range(n_steps):
            # Multi-scale approach: combine different window sizes
            # This preserves both short-term reactions and long-term trends

            # Short-term window (recent changes)
            short_window = max(3, ma_window // 4)
            short_start = max(0, i - short_window + 1)
            short_values = drift_path[short_start : i + 1]

            # Medium-term window (main smoothing)
            med_start = max(0, i - ma_window + 1)
            med_values = drift_path[med_start : i + 1]

            # Long-term window (trend detection)
            long_window = min(ma_window * 2, n_steps)
            long_start = max(0, i - long_window + 1)
            long_values = drift_path[long_start : i + 1]

            # Apply different weights to each scale
            if len(short_values) > 0 and len(med_values) > 0 and len(long_values) > 0:
                # Calculate weighted averages with exponential decay
                alpha_short = 0.3
                weights_short = np.array(
                    [
                        (1 - alpha_short) ** j
                        for j in range(len(short_values) - 1, -1, -1)
                    ]
                )
                weights_short = weights_short / np.sum(weights_short)
                short_avg = np.sum(short_values * weights_short)

                alpha_med = 0.1
                weights_med = np.array(
                    [(1 - alpha_med) ** j for j in range(len(med_values) - 1, -1, -1)]
                )
                weights_med = weights_med / np.sum(weights_med)
                med_avg = np.sum(med_values * weights_med)

                alpha_long = 0.05
                weights_long = np.array(
                    [(1 - alpha_long) ** j for j in range(len(long_values) - 1, -1, -1)]
                )
                weights_long = weights_long / np.sum(weights_long)
                long_avg = np.sum(long_values * weights_long)

                # Blend the three scales with adaptive weights
                # More weight to short-term when volatility is high
                if i > 10:
                    recent_drifts = drift_path[max(0, i - 10) : i + 1]
                    drift_volatility = np.std(recent_drifts)
                    vol_factor = np.tanh(drift_volatility * 5)

                    # Adjust weights based on volatility
                    short_weight = 0.5 + 0.3 * vol_factor
                    med_weight = 0.3 - 0.1 * vol_factor
                    long_weight = 0.2 - 0.1 * vol_factor
                else:
                    # Default weights for early steps
                    short_weight, med_weight, long_weight = 0.5, 0.3, 0.2

                # Combine for final smoothed value
                smoothed_drift_path[i] = (
                    short_weight * short_avg
                    + med_weight * med_avg
                    + long_weight * long_avg
                )
            else:
                # Fallback for early steps
                smoothed_drift_path[i] = drift_path[i]

        # Third pass: generate price path using stochastic process
        # Initialize arrays for fractional process
        increments = np.zeros(n_steps)

        # Generate correlated random increments for fractional Brownian motion
        for i in range(n_steps):
            # Generate standard normal random number
            z = np.random.normal(0, 1)

            # For first few steps, use standard Brownian motion
            if i < self.memory_length:
                increments[i] = z
            else:
                # Apply fractional Brownian motion with memory
                memory_weights = np.power(
                    np.arange(1, self.memory_length + 1, 1), self.hurst - 1.5
                )
                memory_weights = memory_weights / np.sum(memory_weights)

                # Blend new innovation with memory of past innovations
                memory_effect = np.sum(
                    memory_weights * increments[i - self.memory_length : i]
                )
                increments[i] = 0.7 * z + 0.3 * memory_effect

        # Generate path step by step
        for i in range(n_steps):
            current_price = price_path[i]
            mu = smoothed_drift_path[i]

            # Calculate mean reversion component
            log_price_ratio = np.log(current_price / self.S_0)
            mean_reversion = -self.mean_reversion_strength * log_price_ratio

            # Effective drift with mean reversion
            effective_drift = mu + mean_reversion

            # Generate step with fractional characteristics
            drift_term = (effective_drift - 0.5 * self.sigma**2) * self.dt
            diffusion_term = self.sigma * np.sqrt(self.dt) * increments[i]

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
        that better capture the behavior of the advanced stochastic process.

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

        # Realized drift (accounting for mean reversion)
        realized_drift = log_returns.mean() / self.dt
        realized_mu = realized_drift + 0.5 * realized_sigma**2

        # Calculate total return
        total_return = (price_path[-1] / price_path[0]) - 1

        # Enhanced validation approach:
        # 1. Check if drift direction is correct
        # 2. Check if magnitude is within reasonable bounds
        if expected_mu > 0:
            mu_valid_direction = price_path[-1] > price_path[0]
            # For positive drift, final price should be higher
        elif expected_mu < 0:
            mu_valid_direction = price_path[-1] < price_path[0]
            # For negative drift, final price should be lower
        else:  # expected_mu == 0
            # For zero drift, we expect minimal change
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

        # Additional statistics

        # Measure mean reversion
        # Correlation between price level and subsequent return
        # Negative correlation indicates mean reversion
        price_levels = price_path[:-1]
        price_level_norm = (price_levels - self.S_0) / self.S_0
        mean_reversion_corr = np.corrcoef(price_level_norm, log_returns)[0, 1]

        # Measure fractal dimension using variance method
        # This estimates the roughness of the price path
        if len(log_returns) > 50:
            # Calculate variance at different time scales
            scales = [1, 2, 4, 8, 16]
            variances = []

            for scale in scales:
                if scale < len(log_returns):
                    # Aggregate returns at this scale
                    agg_returns = np.array(
                        [
                            np.sum(log_returns[i : i + scale])
                            for i in range(0, len(log_returns) - scale + 1, scale)
                        ]
                    )
                    # Calculate variance
                    variances.append(np.var(agg_returns))

            if len(variances) > 1 and len(scales) > 1:
                # Estimate Hurst exponent from log-log plot
                valid_scales = scales[: len(variances)]
                log_scales = np.log(valid_scales)
                log_variances = np.log(variances)

                # Linear regression
                slope, _ = np.polyfit(log_scales, log_variances, 1)

                # Hurst exponent H = slope/2
                estimated_hurst = slope / 2

                # Fractal dimension D = 2 - H
                estimated_fractal_dim = 2 - estimated_hurst
            else:
                estimated_fractal_dim = 1.5  # Default
        else:
            estimated_fractal_dim = 1.5  # Default for short paths

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
            "mean_reversion_corr": mean_reversion_corr,
            "fractal_dimension": estimated_fractal_dim,
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
            # Step 1: Map exposure to probability using approach
            prob = self.exposure_to_probability(exposure)
            results["probabilities"].append(prob)

            # Step 2: Convert probability to drift using approach
            mu = self.compute_mu_from_probability(prob)
            results["mu_values"].append(mu)

            # Step 3: Generate multiple price paths using process
            paths = []
            validations = []

            for j in range(num_paths_per_exposure):
                path = self.generate_price_path(
                    mu=mu,
                    duration_in_seconds=duration_in_seconds,
                    random_seed=random_seed + i * 1000 + j,
                )
                paths.append(path)

                # Validate each path with metrics
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
                "mean_reversion_strength": -np.mean(
                    [v["mean_reversion_corr"] for v in validations]
                ),
                "mean_fractal_dimension": np.mean(
                    [v["fractal_dimension"] for v in validations]
                ),
            }
            results["summary_stats"].append(summary)

        self.results = results
        return results

    def create_comprehensive_visualization(
        self,
        results: Optional[Dict] = None,
        max_paths_to_plot: int = 50,
        figsize: Tuple[int, int] = (20, 16),  # Larger figure for more plots
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create comprehensive visualization of results with additional
        metrics and insights.

        Parameters:
        -----------
        results : dict, optional
            Results from process_exposure_data (uses self.results if None)
        max_paths_to_plot : int
            Maximum number of individual paths to plot per exposure
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure
        """
        if results is None:
            results = self.results

        if not results:
            raise ValueError("No results to plot. Run process_exposure_data first.")

        n_exposures = len(results["exposures"])

        # Create subplots with more rows for additional metrics
        fig = plt.figure(figsize=figsize, dpi=200)

        # Main price path plots
        gs = fig.add_gridspec(
            5, n_exposures, height_ratios=[3, 1, 1, 1, 1], hspace=0.4, wspace=0.3
        )

        # Plot price paths
        for i, (exposure, paths, mean_path, summary) in enumerate(
            zip(
                results["exposures"],
                results["price_paths"],
                results["mean_paths"],
                results["summary_stats"],
            )
        ):
            ax = fig.add_subplot(gs[0, i])

            # Plot individual paths (sample if too many)
            paths_to_plot = paths[:max_paths_to_plot]
            for path in paths_to_plot:
                ax.plot(path, color="tab:blue", alpha=0.1, linewidth=0.5)

            # Plot mean path
            ax.plot(mean_path, color="red", linewidth=2, label="Mean Path")

            # Add horizontal line at starting price
            ax.axhline(
                y=self.S_0,
                color="black",
                linestyle="--",
                alpha=0.5,
                label="Start Price",
            )

            # Formatting
            ax.set_title(
                f"Exposure: {exposure:,.0f}\n"
                f"P={summary['probability']:.3f}, μ={summary['mu']:.4f}\n"
                f"Realized: σ={summary['mean_realized_sigma']:.3f}, "
                f"μ={summary['mean_realized_mu']:.4f}",
                fontsize=10,
            )
            ax.set_xlabel("Time (seconds)")
            if i == 0:
                ax.set_ylabel("Index Price")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # Format y-axis
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style="plain", axis="y")

            # Add return distribution histogram below each price path
            ax_hist = fig.add_subplot(gs[1, i])

            # Calculate returns for all paths
            returns = [(p[-1] / p[0] - 1) * 100 for p in paths]  # Convert to percentage

            # Plot histogram
            ax_hist.hist(
                returns, bins=20, alpha=0.7, color="skyblue", edgecolor="black"
            )
            ax_hist.axvline(
                x=summary["mean_total_return"] * 100,
                color="red",
                linestyle="-",
                linewidth=2,
                label=f"Mean: {summary['mean_total_return'] * 100:.2f}%",
            )
            ax_hist.set_title("Return Distribution", fontsize=9)
            ax_hist.set_xlabel("Return (%)")
            if i == 0:
                ax_hist.set_ylabel("Frequency")
            ax_hist.grid(True, alpha=0.3)
            ax_hist.legend(fontsize=8)

        # Drift validation plot
        ax_val = fig.add_subplot(gs[2, :])
        exposures = results["exposures"]
        mu_rates = [s["mu_validation_rate"] for s in results["summary_stats"]]

        x_pos = np.arange(len(exposures))

        bars = ax_val.bar(
            x_pos, mu_rates, label="μ (Drift) Validation Rate", alpha=0.7, color="red"
        )

        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, mu_rates)):
            height = bar.get_height()
            ax_val.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{rate:.1%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax_val.set_xlabel("Exposure Scenarios")
        ax_val.set_ylabel("Validation Rate")
        ax_val.set_title("Drift Validation Results")
        ax_val.set_xticks(x_pos)
        ax_val.set_xticklabels([f"{e:,.0f}" for e in exposures], rotation=45)
        ax_val.legend()
        ax_val.grid(True, alpha=0.3)
        ax_val.set_ylim(0, 1.1)

        # Parameter mapping plot
        ax_param = fig.add_subplot(gs[3, :])
        probabilities = results["probabilities"]
        mu_values = results["mu_values"]

        ax_param_twin = ax_param.twinx()

        line1 = ax_param.plot(
            exposures, probabilities, "o-", color="blue", label="Probability"
        )
        line2 = ax_param_twin.plot(exposures, mu_values, "s-", color="red", label="μ")

        ax_param.set_xlabel("Net Exposure")
        ax_param.set_ylabel("Probability", color="blue")
        ax_param_twin.set_ylabel("Drift μ", color="red")
        ax_param.set_title("Exposure → Probability → Drift Mapping")
        ax_param.grid(True, alpha=0.3)

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax_param.legend(lines, labels, loc="upper left")

        # Enhanced metrics plot
        ax_metrics = fig.add_subplot(gs[4, :])

        # Plot fractal dimension and mean reversion strength
        fractal_dim = [s["mean_fractal_dimension"] for s in results["summary_stats"]]
        mean_reversion = [
            s["mean_reversion_strength"] for s in results["summary_stats"]
        ]

        ax_metrics_twin = ax_metrics.twinx()

        line3 = ax_metrics.plot(
            exposures, fractal_dim, "o-", color="purple", label="Fractal Dimension"
        )
        line4 = ax_metrics_twin.plot(
            exposures, mean_reversion, "s-", color="green", label="Mean Reversion"
        )

        ax_metrics.set_xlabel("Net Exposure")
        ax_metrics.set_ylabel("Fractal Dimension", color="purple")
        ax_metrics_twin.set_ylabel("Mean Reversion Strength", color="green")
        ax_metrics.set_title("Creative Process Metrics")
        ax_metrics.grid(True, alpha=0.3)

        # Combined legend
        lines = line3 + line4
        labels = [l.get_label() for l in lines]
        ax_metrics.legend(lines, labels, loc="upper left")

        # Overall title
        fig.suptitle(
            f"Creative Supply-Demand Index Analysis\n"
            f"σ={self.sigma:.1%}, Scale={self.scale:,.0f}, k={self.k:.2f}, "
            f"T={self.T * 365 * 24:.1f}h, MR={self.mean_reversion_strength:.2f}, "
            f"Fractal={self.fractal_dimension:.2f}",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def generate_summary_report(self, results: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate a summary report of all results with metrics

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
            "mean_reversion_strength",
            "mean_fractal_dimension",
        ]:
            df_display[col] = df_display[col].round(3)
        for col in ["mu_percentage", "return_percentage"]:
            df_display[col] = df_display[col].round(2)
        for col in ["exposure", "mean_final_price"]:
            df_display[col] = df_display[col].round(1)

        return df_display

    def print_validation_summary(self, results: Optional[Dict] = None) -> None:
        """
        Print a summary of validation results with creative metrics

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
        print("CREATIVE SUPPLY-DEMAND INDEX VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Engine Parameters:")
        print(f"  σ (volatility): {self.sigma:.1%}")
        print(f"  Scale: {self.scale:,.0f}")
        print(f"  k (max deviation): {self.k:.2f}")
        print(f"  T (time horizon): {self.T * 365 * 24:.1f} hours")
        print(f"  S₀ (initial price): {self.S_0:,.0f}")
        print(f"  Mean Reversion: {self.mean_reversion_strength:.2f}")
        print(f"  Fractal Dimension: {self.fractal_dimension:.2f}")
        print(f"  Smoothness Factor: {self.smoothness_factor:.1f}")
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
            print(f"  Fractal dim: {stats['mean_fractal_dimension']:>8.2f}")
            print(f"  Mean reversion: {stats['mean_reversion_strength']:>6.3f}")
            print()

        # Overall validation rate (drift only)
        overall_mu_rate = np.mean([s["mu_validation_rate"] for s in summary_stats])

        print("-" * 60)
        print(f"Overall Validation Rate:")
        print(f"  μ (drift): {overall_mu_rate:.1%}")
        print(
            "  Note: Creative process includes mean reversion and fractal characteristics"
        )
        print("=" * 60)


def run_example_analysis():
    """
    Example usage of the Supply-Demand Index Engine
    """
    print("Running Supply-Demand Index Example Analysis...")

    # Initialize engine with default parameters
    engine = SupplyDemandIndexEngine(
        sigma=0.1,  # 10% volatility
        scale=15_000,  # Exposure sensitivity
        k=0.4,  # Max probability deviation
        T=1 / (365 * 24),  # 1 hour time horizon
        S_0=100_000,  # Starting price
        mean_reversion_strength=0.05,  # Moderate mean reversion
        fractal_dimension=1.6,  # Slightly rougher than standard Brownian
        memory_length=20,  # Memory length for fractional process
        smoothness_factor=2.0,  # Controls smoothness of transitions
    )

    # Define exposure scenarios
    exposure_scenarios = [-30_000, -15_000, 0, 15_000, 30_000]

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

    # Create visualization
    print("\nGenerating comprehensive visualization...")
    engine.create_comprehensive_visualization(
        save_path="creative_supply_demand_analysis.png"
    )

    print("Analysis complete!")
    return engine, results


if __name__ == "__main__":
    # Run example analysis
    engine, results = run_example_analysis()
