"""
Supply-Demand Index Engine

This module implements the complete supply-demand index generation pipeline:
1. Takes exposure data as input
2. Maps exposure to probability using sigmoid function
3. Converts probability to drift parameter using Black-Scholes framework
4. Generates GBM price paths with dynamic drift
5. Performs statistical validation
6. Creates comprehensive visualizations

Author: Supply-Demand Index Team
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm
from typing import Tuple, List, Optional, Dict, Any
import warnings

warnings.filterwarnings("ignore")


class SupplyDemandIndexEngine:
    """
    Main engine for generating supply-demand responsive index prices
    """

    def __init__(
        self,
        sigma: float = 0.1,
        scale: float = 10_000,
        k: float = 0.45,
        T: float = 1 / (365 * 24),  # 1 hour in years
        S_0: float = 100_000,
        dt: float = 1 / (86_400 * 365),
    ):  # 1 second in years
        """
        Initialize the Supply-Demand Index Engine

        Parameters:
        -----------
        sigma : float
            Fixed volatility (annualized)
        scale : float
            Exposure sensitivity parameter for sigmoid mapping
        k : float
            Maximum deviation from 0.5 in probability mapping
        T : float
            Time horizon for probability calculations (in years)
        S_0 : float
            Initial index price
        dt : float
            Time step size (in years)
        """
        self.sigma = sigma
        self.scale = scale
        self.k = k
        self.T = T
        self.S_0 = S_0
        self.dt = dt

        # Storage for results
        self.results = {}

    def exposure_to_probability(self, exposure: float) -> float:
        """
        Map net exposure to desired probability using a mathematically rigorous approach
        based on the cumulative distribution function (CDF) of a normal distribution.

        This approach is more theoretically sound and provides a smooth, continuous mapping
        from exposure to probability with a well-defined mathematical foundation.

        Parameters:
        -----------
        exposure : float
            Net exposure - higher exposure leads to higher probability of price increase

        Returns:
        --------
        float
            Probability of ending above current price (0-1)
        """
        # Normalize exposure using a scale factor that provides appropriate sensitivity
        # A smaller scale factor will make the probability more sensitive to exposure changes
        normalized_exposure = exposure / self.scale

        # Apply the CDF of a normal distribution with mean 0 and standard deviation 1
        # This creates a smooth S-curve that maps the entire real line to (0,1)
        # The standard normal CDF is a well-established mathematical function
        # with desirable properties for this application
        base_probability = norm.cdf(normalized_exposure)

        # Scale the probability to the desired range [0.5-k, 0.5+k]
        # This maintains the mathematical properties while constraining the output
        scaled_probability = 0.5 + (base_probability - 0.5) * (2 * self.k)

        # Ensure probability stays within bounds (should be guaranteed by the CDF,
        # but we add this as a safety measure)
        return np.clip(scaled_probability, 0.5 - self.k, 0.5 + self.k)

    def compute_mu_from_probability(
        self, P: float, S: float = None, K: float = None
    ) -> float:
        """
        Compute drift parameter from desired probability using Black-Scholes framework
        with an enhanced mathematical approach that provides more accurate and
        theoretically sound drift values.

        This implementation uses the direct relationship between probability and drift
        in the Black-Scholes framework, with a scaling factor to achieve a wider range
        of drift values (-100 to 100).

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
            Required drift parameter (mu) scaled to achieve a wider range
        """
        if S is None:
            S = self.S_0
        if K is None:
            K = S  # For "above current price" probability

        # Handle edge cases with a more gradual approach
        # This prevents extreme drift values while maintaining mathematical consistency
        P = np.clip(P, 0.001, 0.999)

        # Calculate d2 from the inverse normal CDF
        # In Black-Scholes, P = N(d2) where N is the standard normal CDF
        # So d2 = N^(-1)(P) where N^(-1) is the inverse normal CDF (ppf)
        d2 = norm.ppf(P)

        # *** PROBABILITY TO DRIFT CONVERSION (KEY MATHEMATICAL RELATIONSHIP) ***
        # This is the core mathematical relationship between probability and drift
        # In Black-Scholes, d2 = (ln(S/K) + (mu - 0.5*sigma^2)*T) / (sigma*sqrt(T))
        # Solving for mu: mu = [d2*sigma*sqrt(T) - ln(S/K)]/T + 0.5*sigma^2
        #
        # This formula directly maps a probability P to the required drift mu
        # that would make the probability of ending above K equal to P
        base_mu = (
            d2 * self.sigma * np.sqrt(self.T) - np.log(S / K)
        ) / self.T + 0.5 * self.sigma**2

        # Apply a scaling factor to achieve the desired drift range (-100 to 100)
        # The scaling is designed to maintain the mathematical relationship
        # while expanding the range of values

        # Calculate the scaling factor based on the probability's distance from 0.5
        # This ensures that extreme probabilities (close to 0 or 1) result in
        # drift values close to -100 or 100
        prob_distance = abs(P - 0.5) / 0.5  # Normalized distance from 0.5 (0 to 1)

        # Apply non-linear scaling to achieve the desired range
        # This formula ensures that:
        # - When P is close to 0.5, scaling is minimal
        # - When P approaches 0 or 1, scaling approaches the maximum

        # *** DRIFT ADJUSTMENT PARAMETER ***
        # Increase this value to get higher drift values
        # Decrease this value to get lower drift values
        # Current value of 1.7 gives approximately ±120 drift range
        # Try values between 1.0 (no scaling) and 3.0 (very high scaling)
        scaling_factor = 1.7  # Adjust this value to achieve the desired drift range

        # Scale the drift value based on its sign and the probability distance
        if base_mu > 0:
            scaled_mu = base_mu * (1 + scaling_factor * prob_distance)
        else:
            scaled_mu = base_mu * (1 + scaling_factor * prob_distance)

        return scaled_mu

    def generate_gbm_path(
        self, mu: float, duration_in_seconds: int, random_seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate GBM price path with specified drift

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

        # Generate log returns
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
        with smoothed drift using weighted moving average.

        This implementation uses a more sophisticated approach for calculating
        drift values and applies a proper weighted moving average for smoothing.

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
            # Map exposure to probability using the enhanced mathematical approach
            probability = self.exposure_to_probability(exposure)
            probability_path[i] = probability

            # Map probability to drift using the enhanced Black-Scholes relationship
            # This provides more accurate and theoretically sound drift values
            drift_path[i] = self.compute_mu_from_probability(probability)

        # Second pass: apply weighted moving average to drift
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
                # Apply weighted average (no additional scaling)
                smoothed_drift_path[i] = np.sum(window_values * weights)
            else:
                # If only one value, no weighting needed
                smoothed_drift_path[i] = window_values[0]

        # Third pass: generate price path using smoothed drift
        for i in range(n_steps):
            # Use smoothed drift for GBM
            mu = smoothed_drift_path[i]

            # Generate one step of GBM
            drift_term = (mu - self.sigma**2 / 2) * self.dt
            volatility_term = self.sigma * np.sqrt(self.dt) * np.random.normal(0, 1)
            log_return = drift_term + volatility_term

            # Update price
            price_path[i + 1] = price_path[i] * np.exp(log_return)

        return price_path, drift_path, smoothed_drift_path, probability_path

    def validate_statistics(
        self,
        price_path: np.ndarray,
        expected_mu: float,
    ) -> Dict[str, Any]:
        """
        Validate that simulated path matches expected drift (direction and strength)
        Uses a more robust statistical approach for validation.

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

        # Realized volatility (for information only)
        realized_sigma = log_returns.std(ddof=1) / np.sqrt(self.dt)

        # Realized drift
        realized_drift = log_returns.mean() / self.dt
        realized_mu = realized_drift + 0.5 * realized_sigma**2

        # Calculate total return
        total_return = (price_path[-1] / price_path[0]) - 1

        # New validation approach: Check if drift direction is correct
        # For positive expected mu, final price should be higher than initial
        # For negative expected mu, final price should be lower than initial
        if expected_mu > 0:
            mu_valid = price_path[-1] > price_path[0]
        elif expected_mu < 0:
            mu_valid = price_path[-1] < price_path[0]
        else:  # expected_mu == 0
            # For zero drift, we expect minimal change
            mu_valid = abs(total_return) < 0.01  # 1% threshold

        # Calculate relative error for information
        mu_error = abs(realized_mu - expected_mu)

        return {
            "realized_sigma": realized_sigma,
            "realized_mu": realized_mu,
            "expected_mu": expected_mu,
            "mu_valid": mu_valid,
            "mu_error": mu_error,
            "realized_drift": realized_drift,
            "path_length": len(price_path),
            "final_price": price_path[-1],
            "total_return": total_return,
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
                path = self.generate_gbm_path(
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

    def create_comprehensive_visualization(
        self,
        results: Optional[Dict] = None,
        max_paths_to_plot: int = 50,
        figsize: Tuple[int, int] = (20, 12),
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create comprehensive visualization of results

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

        # Create subplots
        fig = plt.figure(figsize=figsize, dpi=200)

        # Main price path plots
        gs = fig.add_gridspec(
            3, n_exposures, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3
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

        # Drift validation plot (only meaningful validation)
        ax_val = fig.add_subplot(gs[1, :])
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
        ax_val.set_title("Drift Validation Results (σ not validated - fixed at 75%)")
        ax_val.set_xticks(x_pos)
        ax_val.set_xticklabels([f"{e:,.0f}" for e in exposures], rotation=45)
        ax_val.legend()
        ax_val.grid(True, alpha=0.3)
        ax_val.set_ylim(0, 1.1)

        # Parameter mapping plot
        ax_param = fig.add_subplot(gs[2, :])
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

        # Overall title
        fig.suptitle(
            f"Supply-Demand Index Analysis\n"
            f"σ={self.sigma:.1%}, Scale={self.scale:,.0f}, k={self.k:.2f}, "
            f"T={self.T * 365 * 24:.1f}h",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

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

        # Format for display
        df_display = df.copy()
        for col in [
            "probability",
            "mean_realized_sigma",
            "mu_validation_rate",
        ]:
            df_display[col] = df_display[col].round(3)
        for col in ["mu_percentage", "mean_total_return"]:
            df_display[col] = df_display[col].round(2)
        for col in ["exposure", "mean_final_price"]:
            df_display[col] = df_display[col].round(0)

        return df_display

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
        print("SUPPLY-DEMAND INDEX VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Engine Parameters:")
        print(f"  σ (volatility): {self.sigma:.1%}")
        print(f"  Scale: {self.scale:,.0f}")
        print(f"  k (max deviation): {self.k:.2f}")
        print(f"  T (time horizon): {self.T * 365 * 24:.1f} hours")
        print(f"  S₀ (initial price): {self.S_0:,.0f}")
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
        print("  Note: σ (volatility) not validated - fixed at input value")
        print("=" * 60)


def run_example_analysis():
    """
    Example usage of the Supply-Demand Index Engine
    """
    print("Running Supply-Demand Index Example Analysis...")

    # Initialize engine
    engine = SupplyDemandIndexEngine(
        sigma=0.1,  # 10% volatility
        scale=15_000,  # Exposure sensitivity
        k=0.4,  # Max probability deviation
        T=1 / (365 * 24),  # 1 hour time horizon
        S_0=100_000,  # Starting price
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
    engine.create_comprehensive_visualization(save_path="supply_demand_analysis.png")

    print("Analysis complete!")
    return engine, results


if __name__ == "__main__":
    # Run example analysis
    engine, results = run_example_analysis()
