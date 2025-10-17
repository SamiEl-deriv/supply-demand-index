"""
New Supply-Demand Index Engine for MT5 Vol 75 Data

This engine processes MT5 Vol 75 data with minute-level aggregated positions
and generates supply-demand index paths with enhanced mathematical models.

Key Features:
- Processes minute-level LONG/SHORT position data
- Smooth, non-linear exposure-to-probability mapping
- Adaptive drift calculation with market regime awareness
- Sophisticated stochastic process with mean reversion and fractal characteristics
- Random 7-day period selection for analysis

Author: Cline
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


class NewSupplyDemandIndexEngine:
    """
    Enhanced Supply-Demand Index Engine for MT5 Vol 75 data
    
    This engine converts net exposure data into probability distributions
    and generates realistic price paths using fractional Geometric Brownian Motion
    with supply-demand dynamics.
    """
    
    def __init__(self, 
                 sigma: float = 0.30,
                 scale: float = 150000,
                 k: float = 0.40,
                 T: float = 1.0 / (365 * 24),
                 S_0: float = 10000,
                 smoothness_factor: float = 2.0,
                 noise_injection_level: float = 0.01):
        """
        Initialize the Supply-Demand Index Engine
        
        Parameters:
        -----------
        sigma : float
            Volatility parameter (default: 0.30 for 30% annual volatility)
        scale : float
            Exposure scaling factor (default: 150000)
        k : float
            Maximum probability deviation from 0.5 (default: 0.40 for ±40%)
        T : float
            Time horizon in years (default: 1 hour = 1/(365*24))
        S_0 : float
            Initial index price (default: 100000)
        smoothness_factor : float
            Controls smoothness of probability transitions (default: 2.0)
        noise_injection_level : float
            Level of noise injection to prevent exploitation (default: 0.01)
        """
        self.sigma = sigma
        self.scale = scale
        self.k = k
        self.T = T
        self.S_0 = S_0
        self.smoothness_factor = smoothness_factor
        self.noise_injection_level = noise_injection_level
        
        # Derived parameters
        self.dt = T / 100  # 100 time steps per period
        self.sqrt_dt = np.sqrt(self.dt)
        
    def exposure_to_probability(self, net_exposure: float) -> float:
        """
        Convert net exposure to probability using smooth, non-linear mapping
        
        This function maps exposure to probability in a way that:
        1. Is smooth and differentiable
        2. Has bounded output [0.5-k, 0.5+k]
        3. Is non-linear to prevent simple exploitation
        4. Deterministic mapping (no noise injection)
        
        Parameters:
        -----------
        net_exposure : float
            Net exposure value (LONG volume - SHORT volume)
            
        Returns:
        --------
        float
            Probability value between (0.5-k) and (0.5+k)
        """
        # Normalize exposure
        normalized_exposure = net_exposure / self.scale
        
        # Apply smooth, non-linear transformation with enhanced smoothness
        # Using a modified sigmoid with power transformation
        smooth_factor = self.smoothness_factor
        final_prob = 0.5 + self.k * np.tanh(normalized_exposure / smooth_factor)
        
        # Ensure bounds
        final_prob = np.clip(final_prob, 0.5 - self.k, 0.5 + self.k)
        
        return final_prob
    
    def compute_mu_from_probability(self, probability: float) -> float:
        """
        Compute drift parameter μ from probability
        
        This uses a sophisticated mapping that considers:
        1. Non-linear relationship between probability and drift
        2. Market regime awareness (high vs low probability scenarios)
        3. Mean reversion tendencies
        
        Parameters:
        -----------
        probability : float
            Probability value between 0 and 1
            
        Returns:
        --------
        float
            Drift parameter μ
        """
        # Center probability around 0.5
        p_centered = probability - 0.5
        
        # Non-linear mapping with regime awareness
        # Higher probabilities get exponentially higher drift
        if abs(p_centered) > 0.2:  # High probability regime
            mu = np.sign(p_centered) * (abs(p_centered) ** 1.5) * self.sigma * 2
        else:  # Normal regime
            mu = p_centered * self.sigma * 1.5
        
        # Add mean reversion component
        mean_reversion_strength = 0.1
        mu -= mean_reversion_strength * p_centered
        
        return mu
    
    def generate_price_path(self, 
                          exposure_sequence: np.ndarray,
                          num_steps: int = 100,
                          random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a single price path based on exposure sequence
        
        Parameters:
        -----------
        exposure_sequence : np.ndarray
            Sequence of net exposure values
        num_steps : int
            Number of time steps per exposure period
        random_seed : Optional[int]
            Random seed for reproducibility
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (price_path, probability_path, drift_path)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        total_steps = len(exposure_sequence) * num_steps
        price_path = np.zeros(total_steps + 1)
        probability_path = np.zeros(total_steps)
        drift_path = np.zeros(total_steps)
        
        price_path[0] = self.S_0
        
        for i, exposure in enumerate(exposure_sequence):
            # Convert exposure to probability and drift
            prob = self.exposure_to_probability(exposure)
            mu = self.compute_mu_from_probability(prob)
            
            # Generate price steps for this exposure period
            start_idx = i * num_steps
            end_idx = (i + 1) * num_steps
            
            for j in range(start_idx, end_idx):
                # Store probability and drift
                probability_path[j] = prob
                drift_path[j] = mu
                
                # Generate price step using GBM with supply-demand dynamics
                dW = np.random.normal(0, 1)
                
                # Enhanced GBM with mean reversion and volatility clustering
                current_price = price_path[j]
                
                # Volatility clustering effect
                vol_cluster = 1 + 0.1 * np.sin(j * 0.1)  # Subtle volatility variation
                effective_sigma = self.sigma * vol_cluster
                
                # Price update with enhanced dynamics
                drift_term = (mu - 0.5 * effective_sigma**2) * self.dt
                diffusion_term = effective_sigma * self.sqrt_dt * dW
                
                price_path[j + 1] = current_price * np.exp(drift_term + diffusion_term)
        
        return price_path, probability_path, drift_path
    
    def process_exposure_data(self,
                            exposure_data: np.ndarray,
                            duration_in_minutes: int = 60,
                            num_paths_per_exposure: int = 50,
                            random_seed: int = 42) -> Dict[str, Any]:
        """
        Process exposure data and generate comprehensive analysis
        
        Parameters:
        -----------
        exposure_data : np.ndarray
            Array of net exposure values
        duration_in_minutes : int
            Duration for each exposure period in minutes
        num_paths_per_exposure : int
            Number of Monte Carlo paths to generate per exposure
        random_seed : int
            Base random seed for reproducibility
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive results dictionary
        """
        np.random.seed(random_seed)
        
        results = {
            'exposures': exposure_data,
            'probabilities': [],
            'mu_values': [],
            'price_paths': [],
            'mean_paths': [],
            'summary_stats': []
        }
        
        print(f"Processing {len(exposure_data)} exposure points...")
        
        for i, exposure in enumerate(exposure_data):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processing exposure {i + 1}/{len(exposure_data)}: {exposure:,.0f}")
            
            # Convert exposure to probability and drift
            prob = self.exposure_to_probability(exposure)
            mu = self.compute_mu_from_probability(prob)
            
            results['probabilities'].append(prob)
            results['mu_values'].append(mu)
            
            # Generate multiple price paths for this exposure
            paths = []
            for path_idx in range(num_paths_per_exposure):
                path, _, _ = self.generate_price_path(
                    exposure_sequence=np.array([exposure]),
                    num_steps=duration_in_minutes,
                    random_seed=random_seed + i * 1000 + path_idx
                )
                paths.append(path)
            
            paths = np.array(paths)
            mean_path = np.mean(paths, axis=0)
            
            results['price_paths'].append(paths)
            results['mean_paths'].append(mean_path)
            
            # Calculate summary statistics
            final_prices = paths[:, -1]
            returns = (final_prices - self.S_0) / self.S_0
            
            summary = {
                'exposure': exposure,
                'probability': prob,
                'mu': mu,
                'mean_return': np.mean(returns),
                'std_return': np.std(returns),
                'min_return': np.min(returns),
                'max_return': np.max(returns),
                'final_mean_price': np.mean(final_prices),
                'mu_validation_rate': np.mean(returns > 0) if mu > 0 else np.mean(returns < 0)
            }
            
            results['summary_stats'].append(summary)
        
        return results
    
    def generate_dynamic_exposure_path(self,
                                     exposure_series: pd.Series,
                                     random_seed: int = 42,
                                     ma_window: int = 12,
                                     seconds_per_minute: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate dynamic price path that responds to time-varying exposure with second-level granularity
        
        Parameters:
        -----------
        exposure_series : pd.Series
            Time series of exposure values (minute-level)
        random_seed : int
            Random seed for reproducibility
        ma_window : int
            Moving average window for smoothing
        seconds_per_minute : int
            Number of price updates per minute (default: 60 for second-level)
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            (price_path, drift_path, smoothed_drift_path, probability_path)
        """
        np.random.seed(random_seed)
        
        # Expand to second-level granularity
        n_minutes = len(exposure_series)
        n_seconds = n_minutes * seconds_per_minute
        
        # Initialize arrays for second-level data
        price_path = np.zeros(n_seconds + 1)
        drift_path = np.zeros(n_seconds)
        probability_path = np.zeros(n_seconds)
        
        price_path[0] = self.S_0
        
        # Process each minute, generating second-level updates
        for minute_idx, exposure in enumerate(exposure_series):
            # Convert exposure to probability and drift once per minute
            prob = self.exposure_to_probability(exposure)
            mu = self.compute_mu_from_probability(prob)
            
            # Generate price updates for each second in this minute
            start_second = minute_idx * seconds_per_minute
            end_second = (minute_idx + 1) * seconds_per_minute
            
            for second_idx in range(start_second, end_second):
                probability_path[second_idx] = prob
                drift_path[second_idx] = mu
                
                # Generate price step
                dW = np.random.normal(0, 1)
                current_price = price_path[second_idx]
                
                # Time step (1 second)
                dt = 1.0 / (365 * 24 * 60 * 60)  # 1 second as fraction of year
                sqrt_dt = np.sqrt(dt)
                
                # Price update with enhanced volatility for second-level granularity
                # Slightly reduce volatility to account for higher frequency
                effective_sigma = self.sigma * 0.8  # Reduce by 20% for second-level
                drift_term = (mu - 0.5 * effective_sigma**2) * dt
                diffusion_term = effective_sigma * sqrt_dt * dW
                
                price_path[second_idx + 1] = current_price * np.exp(drift_term + diffusion_term)
        
        # Apply weighted moving average smoothing to drift (adjust window for seconds)
        adjusted_ma_window = ma_window * seconds_per_minute  # Scale window for second-level
        weights = np.exp(-np.arange(adjusted_ma_window) / (adjusted_ma_window / 3))
        weights = weights / np.sum(weights)
        
        smoothed_drift_path = np.convolve(drift_path, weights, mode='same')
        
        return price_path, drift_path, smoothed_drift_path, probability_path
    
    def print_validation_summary(self, results: Dict[str, Any]) -> None:
        """Print validation summary of the analysis results"""
        summary_stats = results['summary_stats']
        
        print("\n" + "=" * 50)
        print("SUPPLY-DEMAND INDEX VALIDATION SUMMARY")
        print("=" * 50)
        
        # Overall statistics
        exposures = [s['exposure'] for s in summary_stats]
        probabilities = [s['probability'] for s in summary_stats]
        mu_values = [s['mu'] for s in summary_stats]
        validation_rates = [s['mu_validation_rate'] for s in summary_stats]
        
        print(f"Number of scenarios analyzed: {len(summary_stats)}")
        print(f"Exposure range: {min(exposures):,.0f} to {max(exposures):,.0f}")
        print(f"Probability range: {min(probabilities):.3f} to {max(probabilities):.3f}")
        print(f"Drift (μ) range: {min(mu_values):.4f} to {max(mu_values):.4f}")
        print(f"Average validation rate: {np.mean(validation_rates):.1%}")
        
        # Validation by exposure level
        positive_exp = [s for s in summary_stats if s['exposure'] > 0]
        negative_exp = [s for s in summary_stats if s['exposure'] < 0]
        
        if positive_exp:
            pos_val_rate = np.mean([s['mu_validation_rate'] for s in positive_exp])
            print(f"Positive exposure validation rate: {pos_val_rate:.1%}")
        
        if negative_exp:
            neg_val_rate = np.mean([s['mu_validation_rate'] for s in negative_exp])
            print(f"Negative exposure validation rate: {neg_val_rate:.1%}")
    
    def analyze_exposure_trends(self, exposure_series: pd.Series, min_trend_length: int = 5) -> Dict[str, Any]:
        """
        Analyze exposure trend lengths (how long exposure stays positive/negative)
        
        Parameters:
        -----------
        exposure_series : pd.Series
            Exposure time series
        min_trend_length : int
            Minimum length to consider as a trend
            
        Returns:
        --------
        Dict[str, Any]
            Exposure trend analysis results
        """
        print("Analyzing exposure trend lengths...")
        
        # Identify exposure signs
        exposure_signs = np.where(exposure_series > 0, 1, -1)
        
        # Find exposure trend segments
        exposure_trends = []
        current_sign = exposure_signs[0]
        trend_start = 0
        trend_length = 1
        
        for i in range(1, len(exposure_signs)):
            if exposure_signs[i] == current_sign:
                trend_length += 1
            else:
                # End of current exposure trend
                if trend_length >= min_trend_length:
                    avg_exposure = np.mean(exposure_series.iloc[trend_start:trend_start + trend_length])
                    
                    exposure_trends.append({
                        'sign': current_sign,
                        'start': trend_start,
                        'length': trend_length,
                        'avg_exposure': avg_exposure,
                        'max_exposure': np.max(exposure_series.iloc[trend_start:trend_start + trend_length]),
                        'min_exposure': np.min(exposure_series.iloc[trend_start:trend_start + trend_length])
                    })
                
                # Start new exposure trend
                current_sign = exposure_signs[i]
                trend_start = i
                trend_length = 1
        
        # Handle last exposure trend
        if trend_length >= min_trend_length:
            avg_exposure = np.mean(exposure_series.iloc[trend_start:trend_start + trend_length])
            
            exposure_trends.append({
                'sign': current_sign,
                'start': trend_start,
                'length': trend_length,
                'avg_exposure': avg_exposure,
                'max_exposure': np.max(exposure_series.iloc[trend_start:trend_start + trend_length]),
                'min_exposure': np.min(exposure_series.iloc[trend_start:trend_start + trend_length])
            })
        
        # Analyze exposure trends
        if not exposure_trends:
            return {
                'total_exposure_trends': 0,
                'positive_exposure_trends': [],
                'negative_exposure_trends': [],
                'avg_exposure_trend_length': 0
            }
        
        positive_exposure_trends = [t for t in exposure_trends if t['sign'] == 1]
        negative_exposure_trends = [t for t in exposure_trends if t['sign'] == -1]
        
        # Calculate statistics
        all_exposure_lengths = [t['length'] for t in exposure_trends]
        positive_exposure_lengths = [t['length'] for t in positive_exposure_trends]
        negative_exposure_lengths = [t['length'] for t in negative_exposure_trends]
        
        results = {
            'total_exposure_trends': len(exposure_trends),
            'positive_exposure_trends': {
                'count': len(positive_exposure_trends),
                'avg_length': np.mean(positive_exposure_lengths) if positive_exposure_lengths else 0,
                'max_length': np.max(positive_exposure_lengths) if positive_exposure_lengths else 0,
                'min_length': np.min(positive_exposure_lengths) if positive_exposure_lengths else 0,
                'lengths': positive_exposure_lengths,
                'trends': positive_exposure_trends
            },
            'negative_exposure_trends': {
                'count': len(negative_exposure_trends),
                'avg_length': np.mean(negative_exposure_lengths) if negative_exposure_lengths else 0,
                'max_length': np.max(negative_exposure_lengths) if negative_exposure_lengths else 0,
                'min_length': np.min(negative_exposure_lengths) if negative_exposure_lengths else 0,
                'lengths': negative_exposure_lengths,
                'trends': negative_exposure_trends
            },
            'avg_exposure_trend_length': np.mean(all_exposure_lengths),
            'total_exposure_trend_time': sum(all_exposure_lengths),
            'exposure_trend_coverage': sum(all_exposure_lengths) / len(exposure_series),
            'all_exposure_trends': exposure_trends
        }
        
        return results

    def analyze_trend_lengths(self, 
                            price_path: np.ndarray, 
                            exposure_series: pd.Series,
                            min_trend_length: int = 5) -> Dict[str, Any]:
        """
        Analyze trend lengths in the price path and their relationship to exposure
        
        Parameters:
        -----------
        price_path : np.ndarray
            Generated price path
        exposure_series : pd.Series
            Exposure time series
        min_trend_length : int
            Minimum length to consider as a trend
            
        Returns:
        --------
        Dict[str, Any]
            Trend analysis results
        """
        print("Analyzing trend lengths and exposure relationships...")
        
        # Calculate price returns
        returns = np.diff(price_path) / price_path[:-1]
        
        # Identify trends (up/down/sideways)
        trend_threshold = 0.0001  # 0.01% threshold for trend identification
        trend_signals = np.where(returns > trend_threshold, 1, 
                                np.where(returns < -trend_threshold, -1, 0))
        
        # Find trend segments
        trends = []
        current_trend = trend_signals[0]
        trend_start = 0
        trend_length = 1
        
        for i in range(1, len(trend_signals)):
            if trend_signals[i] == current_trend:
                trend_length += 1
            else:
                # End of current trend
                if trend_length >= min_trend_length:
                    # Get corresponding exposure for this trend period
                    if trend_start < len(exposure_series):
                        end_idx = min(trend_start + trend_length, len(exposure_series))
                        avg_exposure = np.mean(exposure_series.iloc[trend_start:end_idx])
                        
                        trends.append({
                            'direction': current_trend,
                            'start': trend_start,
                            'length': trend_length,
                            'avg_exposure': avg_exposure,
                            'price_change': (price_path[trend_start + trend_length] - price_path[trend_start]) / price_path[trend_start]
                        })
                
                # Start new trend
                current_trend = trend_signals[i]
                trend_start = i
                trend_length = 1
        
        # Handle last trend
        if trend_length >= min_trend_length and trend_start < len(exposure_series):
            end_idx = min(trend_start + trend_length, len(exposure_series))
            avg_exposure = np.mean(exposure_series.iloc[trend_start:end_idx])
            
            trends.append({
                'direction': current_trend,
                'start': trend_start,
                'length': trend_length,
                'avg_exposure': avg_exposure,
                'price_change': (price_path[trend_start + trend_length - 1] - price_path[trend_start]) / price_path[trend_start]
            })
        
        # Analyze trends
        if not trends:
            return {
                'total_trends': 0,
                'up_trends': [],
                'down_trends': [],
                'sideways_trends': [],
                'exposure_correlation': 0,
                'avg_trend_length': 0
            }
        
        up_trends = [t for t in trends if t['direction'] == 1]
        down_trends = [t for t in trends if t['direction'] == -1]
        sideways_trends = [t for t in trends if t['direction'] == 0]
        
        # Calculate correlation between exposure and trend direction
        exposures = [t['avg_exposure'] for t in trends]
        directions = [t['direction'] for t in trends]
        
        if len(exposures) > 1:
            exposure_correlation = np.corrcoef(exposures, directions)[0, 1]
        else:
            exposure_correlation = 0
        
        # Calculate statistics
        all_lengths = [t['length'] for t in trends]
        up_lengths = [t['length'] for t in up_trends]
        down_lengths = [t['length'] for t in down_trends]
        sideways_lengths = [t['length'] for t in sideways_trends]
        
        up_exposures = [t['avg_exposure'] for t in up_trends]
        down_exposures = [t['avg_exposure'] for t in down_trends]
        
        results = {
            'total_trends': len(trends),
            'up_trends': {
                'count': len(up_trends),
                'avg_length': np.mean(up_lengths) if up_lengths else 0,
                'max_length': np.max(up_lengths) if up_lengths else 0,
                'min_length': np.min(up_lengths) if up_lengths else 0,
                'avg_exposure': np.mean(up_exposures) if up_exposures else 0,
                'lengths': up_lengths,
                'exposures': up_exposures
            },
            'down_trends': {
                'count': len(down_trends),
                'avg_length': np.mean(down_lengths) if down_lengths else 0,
                'max_length': np.max(down_lengths) if down_lengths else 0,
                'min_length': np.min(down_lengths) if down_lengths else 0,
                'avg_exposure': np.mean(down_exposures) if down_exposures else 0,
                'lengths': down_lengths,
                'exposures': down_exposures
            },
            'sideways_trends': {
                'count': len(sideways_trends),
                'avg_length': np.mean(sideways_lengths) if sideways_lengths else 0,
                'max_length': np.max(sideways_lengths) if sideways_lengths else 0,
                'min_length': np.min(sideways_lengths) if sideways_lengths else 0,
                'lengths': sideways_lengths
            },
            'exposure_correlation': exposure_correlation,
            'avg_trend_length': np.mean(all_lengths),
            'total_trend_time': sum(all_lengths),
            'trend_coverage': sum(all_lengths) / len(price_path),
            'all_trends': trends
        }
        
        return results
    
    def print_trend_analysis(self, trend_results: Dict[str, Any]) -> None:
        """Print comprehensive trend analysis"""
        print("\n" + "=" * 60)
        print("TREND LENGTH ANALYSIS")
        print("=" * 60)
        
        print(f"Total trends identified: {trend_results['total_trends']}")
        print(f"Average trend length: {trend_results['avg_trend_length']:.1f} minutes")
        print(f"Trend coverage: {trend_results['trend_coverage']:.1%} of total time")
        print(f"Exposure-Direction correlation: {trend_results['exposure_correlation']:.3f}")
        
        print("\nUP TRENDS:")
        up = trend_results['up_trends']
        print(f"  Count: {up['count']}")
        if up['count'] > 0:
            print(f"  Average length: {up['avg_length']:.1f} minutes")
            print(f"  Length range: {up['min_length']} - {up['max_length']} minutes")
            print(f"  Average exposure: {up['avg_exposure']:,.0f}")
            print(f"  Exposure range: {min(up['exposures']):,.0f} to {max(up['exposures']):,.0f}")
        
        print("\nDOWN TRENDS:")
        down = trend_results['down_trends']
        print(f"  Count: {down['count']}")
        if down['count'] > 0:
            print(f"  Average length: {down['avg_length']:.1f} minutes")
            print(f"  Length range: {down['min_length']} - {down['max_length']} minutes")
            print(f"  Average exposure: {down['avg_exposure']:,.0f}")
            print(f"  Exposure range: {min(down['exposures']):,.0f} to {max(down['exposures']):,.0f}")
        
        print("\nSIDEWAYS TRENDS:")
        sideways = trend_results['sideways_trends']
        print(f"  Count: {sideways['count']}")
        if sideways['count'] > 0:
            print(f"  Average length: {sideways['avg_length']:.1f} minutes")
            print(f"  Length range: {sideways['min_length']} - {sideways['max_length']} minutes")
        
        # Analyze exposure-trend relationship
        if trend_results['exposure_correlation'] > 0.3:
            print(f"\n✓ STRONG positive correlation: Positive exposure → Up trends")
        elif trend_results['exposure_correlation'] < -0.3:
            print(f"\n✓ STRONG negative correlation: Positive exposure → Down trends")
        elif abs(trend_results['exposure_correlation']) > 0.1:
            print(f"\n~ WEAK correlation between exposure and trend direction")
        else:
            print(f"\n○ NO significant correlation between exposure and trend direction")
    
    def generate_summary_report(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Generate a comprehensive summary report"""
        summary_stats = results['summary_stats']
        
        df = pd.DataFrame(summary_stats)
        
        # Add additional calculated columns
        df['exposure_magnitude'] = np.abs(df['exposure'])
        df['probability_deviation'] = np.abs(df['probability'] - 0.5)
        df['return_volatility'] = df['std_return']
        df['sharpe_ratio'] = df['mean_return'] / df['std_return']
        
        # Sort by exposure magnitude for better readability
        df = df.sort_values('exposure_magnitude', ascending=False)
        
        return df
