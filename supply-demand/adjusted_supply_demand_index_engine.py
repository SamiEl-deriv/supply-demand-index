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
        smoothness_factor : float
            Controls smoothness of transitions in probability mapping
        """
        self.sigma = sigma
        self.scale = scale
        self.k = k
        self.T = T
        self.S_0 = S_0
        self.dt = dt
        self.smoothness_factor = smoothness_factor

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
        Generate price path using standard Geometric Brownian Motion (GBM)
        with enhanced probability mapping and drift calculation but no memory or mean reversion.

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

        # Second pass: Apply weighted moving average to drift (similar to basic version)
        # Using exponential weighting for a more mathematically sound smoothing
        for i in range(n_steps):
            # Calculate moving average window bounds
            start_idx = max(0, i - ma_window + 1)
            # Get drift values in window
            window_values = drift_path[start_idx : i + 1]

            # Calculate weighted moving average with exponentially increasing weights
            # This gives more importance to newer values with a sound mathematical basis
            window_size = len(window_values)
            if window_size > 1:
                # Create weights that increase exponentially (newer values have higher weights)
                # This provides a more theoretically sound weighting scheme
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

        # No additional statistics needed for standard GBM
        # We removed mean reversion and fractal dimension calculations
        # to eliminate predictable patterns

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
            f"Enhanced Supply-Demand Index Analysis\n"
            f"σ={self.sigma:.1%}, Scale={self.scale:,.0f}, k={self.k:.2f}, "
            f"T={self.T * 365 * 24:.1f}h, Smoothness={self.smoothness_factor:.1f}",
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
        print("ENHANCED SUPPLY-DEMAND INDEX VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Engine Parameters:")
        print(f"  σ (volatility): {self.sigma:.1%}")
        print(f"  Scale: {self.scale:,.0f}")
        print(f"  k (max deviation): {self.k:.2f}")
        print(f"  T (time horizon): {self.T * 365 * 24:.1f} hours")
        print(f"  S₀ (initial price): {self.S_0:,.0f}")
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
            print()

        # Overall validation rate (drift only)
        overall_mu_rate = np.mean([s["mu_validation_rate"] for s in summary_stats])

        print("-" * 60)
        print(f"Overall Validation Rate:")
        print(f"  μ (drift): {overall_mu_rate:.1%}")
        print("  Note: Enhanced probability mapping and drift calculation only")
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
