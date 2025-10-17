"""
Risk-Optimized Supply-Demand Index Engine

This module implements a risk-optimized approach to supply-demand index generation with:
1. Advanced parameter optimization using Bayesian methods
2. Anti-exploitation mechanisms with controlled randomness
3. Comprehensive backtesting framework for parameter validation
4. Real-time risk monitoring and adaptive parameter adjustment
5. Multi-objective optimization balancing responsiveness, exploitability, and risk

Author: Cline
Date: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm, t, skewnorm
from scipy.optimize import minimize
# from sklearn.metrics import mean_squared_error  # Not used, commented out
from typing import Tuple, List, Optional, Dict, Any, Union
import warnings
import time
from datetime import datetime, timedelta
import json

warnings.filterwarnings("ignore")

# Try to import optimization libraries
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False
    print("Warning: scikit-optimize not available. Using scipy optimization instead.")


class RiskOptimizedSupplyDemandIndexEngine:
    """
    Risk-optimized engine for generating supply-demand responsive index prices
    with advanced parameter optimization and anti-exploitation mechanisms
    """

    def __init__(
        self,
        sigma: float = 0.35,
        scale: float = 140_000,
        k: float = 0.38,
        T: float = 1 / (365 * 24),  # 1 hour in years
        S_0: float = 100_000,
        dt: float = 1 / (86_400 * 365),  # 1 second in years
        mean_reversion_strength: float = 0.05,
        fractal_dimension: float = 1.5,
        memory_length: int = 20,
        regime_threshold: float = 0.3,
        smoothness_factor: float = 2.0,
        # Risk optimization parameters
        sigma_noise_std: float = 0.05,
        scale_adaptation_rate: float = 0.1,
        k_oscillation_amplitude: float = 0.05,
        noise_injection_level: float = 0.015,
        pattern_breaking_frequency: float = 0.1,
        psychology_bull_factor: float = 0.35,
        psychology_bear_factor: float = 0.45,
        # Risk constraints
        max_daily_movement: float = 0.05,
        max_volatility: float = 0.8,
        min_smoothing: float = 0.05,
        max_sensitivity: float = 0.001,
    ):
        """
        Initialize the Risk-Optimized Supply-Demand Index Engine

        Parameters:
        -----------
        [Standard parameters from creative engine...]
        sigma_noise_std : float
            Standard deviation for daily sigma noise injection
        scale_adaptation_rate : float
            Rate of adaptation for dynamic scaling
        k_oscillation_amplitude : float
            Amplitude of k parameter oscillation to prevent boundary exploitation
        noise_injection_level : float
            Level of controlled noise injection (as fraction of base value)
        pattern_breaking_frequency : float
            Frequency of extra noise injection for pattern breaking
        psychology_bull_factor : float
            Psychology scaling factor for bullish scenarios
        psychology_bear_factor : float
            Psychology scaling factor for bearish scenarios
        max_daily_movement : float
            Maximum allowed daily movement (risk constraint)
        max_volatility : float
            Maximum allowed volatility (risk constraint)
        min_smoothing : float
            Minimum smoothing level (anti-exploitation constraint)
        max_sensitivity : float
            Maximum sensitivity to single trade (risk constraint)
        """
        # Base parameters
        self.sigma_base = sigma
        self.scale_base = scale
        self.k_base = k
        self.T = T
        self.S_0 = S_0
        self.dt = dt
        self.mean_reversion_strength = mean_reversion_strength
        self.fractal_dimension = fractal_dimension
        self.memory_length = memory_length
        self.regime_threshold = regime_threshold
        self.smoothness_factor = smoothness_factor
        self.hurst = 2 - fractal_dimension

        # Risk optimization parameters
        self.sigma_noise_std = sigma_noise_std
        self.scale_adaptation_rate = scale_adaptation_rate
        self.k_oscillation_amplitude = k_oscillation_amplitude
        self.noise_injection_level = noise_injection_level
        self.pattern_breaking_frequency = pattern_breaking_frequency
        self.psychology_bull_factor = psychology_bull_factor
        self.psychology_bear_factor = psychology_bear_factor

        # Risk constraints
        self.max_daily_movement = max_daily_movement
        self.max_volatility = max_volatility
        self.min_smoothing = min_smoothing
        self.max_sensitivity = max_sensitivity

        # Dynamic parameters (updated during operation)
        self.sigma = sigma
        self.scale = scale
        self.k = k

        # Storage for results and optimization history
        self.results = {}
        self.optimization_history = []
        self.risk_metrics_history = []
        self.parameter_history = []

        # Market regime detection
        self.current_regime = "normal"
        self.regime_history = []

        # Anti-exploitation state
        self.last_parameter_update = datetime.now()
        self.exploitation_attempts = 0

    def add_anti_exploitation_noise(self, base_value: float, noise_level: float = None) -> float:
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
            
        # Add pattern-breaking noise with specified frequency
        if np.random.random() < self.pattern_breaking_frequency:
            noise_level *= 2  # Double noise for pattern breaking
            
        noise = np.random.normal(0, noise_level * abs(base_value))
        return base_value + noise

    def detect_market_regime(self, recent_exposures: np.ndarray, recent_volatility: float) -> str:
        """
        Detect current market regime for parameter adaptation
        
        Parameters:
        -----------
        recent_exposures : np.ndarray
            Recent exposure values
        recent_volatility : float
            Recent volatility estimate
            
        Returns:
        --------
        str
            Market regime: 'normal', 'high_stress', or 'trending'
        """
        volatility_threshold_high = self.sigma_base * 1.5
        exposure_threshold = self.scale_base * 0.5
        
        if recent_volatility > volatility_threshold_high:
            return "high_stress"
        elif abs(np.mean(recent_exposures)) > exposure_threshold:
            return "trending"
        else:
            return "normal"

    def get_regime_parameters(self, regime: str) -> Dict[str, float]:
        """
        Get optimized parameters for specific market regime
        
        Parameters:
        -----------
        regime : str
            Market regime
            
        Returns:
        --------
        Dict[str, float]
            Regime-specific parameters
        """
        params = {
            "normal": {
                "sigma": self.sigma_base,
                "scale": self.scale_base,
                "k": self.k_base
            },
            "high_stress": {
                "sigma": self.sigma_base * 0.7,  # Less volatile
                "scale": self.scale_base * 1.4,  # Less sensitive
                "k": self.k_base * 0.85  # More conservative
            },
            "trending": {
                "sigma": self.sigma_base * 1.3,  # More volatile
                "scale": self.scale_base * 0.8,  # More sensitive
                "k": self.k_base * 1.2  # More responsive
            }
        }
        return params[regime]

    def get_time_adjusted_parameters(self, base_params: Dict[str, float], current_time: datetime) -> Dict[str, float]:
        """
        Adjust parameters based on time of day/week/month
        
        Parameters:
        -----------
        base_params : Dict[str, float]
            Base parameters
        current_time : datetime
            Current timestamp
            
        Returns:
        --------
        Dict[str, float]
            Time-adjusted parameters
        """
        hour = current_time.hour
        day_of_week = current_time.weekday()
        
        # Different behavior during different market sessions
        if 8 <= hour <= 16:  # Main trading hours
            sensitivity_multiplier = 1.0
        elif 16 <= hour <= 20:  # Overlap hours
            sensitivity_multiplier = 1.2  # More responsive
        else:  # Off hours
            sensitivity_multiplier = 0.8  # Less responsive
        
        adjusted_params = base_params.copy()
        adjusted_params['scale'] *= (1/sensitivity_multiplier)  # Inverse relationship
        
        return adjusted_params

    def update_dynamic_parameters(self, current_time: datetime = None, recent_data: Dict = None):
        """
        Update dynamic parameters based on current conditions
        
        Parameters:
        -----------
        current_time : datetime, optional
            Current timestamp
        recent_data : Dict, optional
            Recent market data for regime detection
        """
        if current_time is None:
            current_time = datetime.now()
            
        # Update sigma with controlled noise
        day_of_year = current_time.timetuple().tm_yday
        sigma_noise = self.sigma_noise_std * np.random.normal(0, 1)
        sigma_trend = 0.05 * np.sin(2 * np.pi * day_of_year / 365)
        self.sigma = max(0.2, min(self.max_volatility, 
                                 self.sigma_base + sigma_noise + sigma_trend))
        
        # Update k with oscillation to prevent boundary exploitation
        time_factor = (current_time.timestamp() / 86400) % 1  # Daily cycle
        k_oscillation = self.k_oscillation_amplitude * np.sin(2 * np.pi * time_factor)
        self.k = max(0.1, min(0.5, self.k_base + k_oscillation))
        
        # Update scale based on recent volume if available
        if recent_data and 'recent_volume' in recent_data:
            volume_factor = recent_data['recent_volume'] / recent_data.get('baseline_volume', 1)
            scale_adjustment = self.scale_adaptation_rate * (volume_factor - 1)
            self.scale = max(50000, self.scale_base * (1 + scale_adjustment))
        
        # Detect and adapt to market regime
        if recent_data and 'recent_exposures' in recent_data and 'recent_volatility' in recent_data:
            new_regime = self.detect_market_regime(
                recent_data['recent_exposures'], 
                recent_data['recent_volatility']
            )
            
            if new_regime != self.current_regime:
                self.current_regime = new_regime
                regime_params = self.get_regime_parameters(new_regime)
                
                # Smooth transition to new regime parameters
                transition_rate = 0.1
                self.sigma = (1 - transition_rate) * self.sigma + transition_rate * regime_params['sigma']
                self.scale = (1 - transition_rate) * self.scale + transition_rate * regime_params['scale']
                self.k = (1 - transition_rate) * self.k + transition_rate * regime_params['k']
        
        # Apply time-based adjustments
        time_params = self.get_time_adjusted_parameters(
            {'sigma': self.sigma, 'scale': self.scale, 'k': self.k},
            current_time
        )
        self.sigma = time_params['sigma']
        self.scale = time_params['scale']
        self.k = time_params['k']
        
        # Record parameter history
        self.parameter_history.append({
            'timestamp': current_time,
            'sigma': self.sigma,
            'scale': self.scale,
            'k': self.k,
            'regime': self.current_regime
        })

    def exposure_to_probability(self, exposure: float) -> float:
        """
        Map net exposure to probability with anti-exploitation enhancements
        """
        # Apply dynamic parameter updates
        self.update_dynamic_parameters()
        
        # Normalize exposure with current dynamic scale
        normalized_exposure = exposure / self.scale
        
        # Prevent overflow by capping normalized exposure
        normalized_exposure = np.clip(normalized_exposure, -100, 100)  # Reasonable bounds

        # Apply sigmoid-based transformation with current smoothness
        sigmoid_arg = -self.smoothness_factor * normalized_exposure
        sigmoid_arg = np.clip(sigmoid_arg, -700, 700)  # Prevent overflow in sigmoid
        base_sigmoid = 1 / (1 + np.exp(sigmoid_arg))

        # Apply arctan transformation
        arctan_component = np.arctan(normalized_exposure) / np.pi + 0.5

        # Blend components with adaptive weighting (with overflow protection)
        exp_arg = -0.5 * normalized_exposure**2
        # Prevent overflow by capping the exponent argument
        exp_arg = np.clip(exp_arg, -700, 700)  # e^700 is close to float64 limit
        blend_weight = np.exp(exp_arg)
        blended_probability = (
            blend_weight * base_sigmoid + (1 - blend_weight) * arctan_component
        )

        # Apply enhanced market psychology with regime awareness
        if exposure >= 0:
            psychology_factor = np.tanh(normalized_exposure / 2) * self.psychology_bull_factor
        else:
            psychology_factor = -np.tanh(-normalized_exposure / 1.5) * self.psychology_bear_factor

        # Incorporate market psychology
        adjusted_probability = blended_probability + 0.1 * psychology_factor

        # Scale to desired range with current dynamic k
        scaled_probability = 0.5 + (adjusted_probability - 0.5) * (2 * self.k)

        # Add anti-exploitation noise
        final_probability = self.add_anti_exploitation_noise(scaled_probability)

        # Ensure probability stays within bounds
        return np.clip(final_probability, 0.5 - self.k, 0.5 + self.k)

    def compute_mu_from_probability(self, P: float, S: float = None, K: float = None) -> float:
        """
        Compute drift parameter from probability with anti-exploitation enhancements
        """
        if S is None:
            S = self.S_0
        if K is None:
            K = S

        # Handle edge cases
        P = np.clip(P, 0.001, 0.999)

        # Calculate probability distance from neutral
        prob_distance = P - 0.5
        abs_prob_distance = abs(prob_distance)
        bullish_sentiment = prob_distance > 0

        # Enhanced distribution blending with regime awareness
        extremity_factor = np.tanh(4 * abs_prob_distance)

        # Adjust distribution selection based on current regime
        if self.current_regime == "high_stress":
            extremity_factor *= 1.5  # More likely to use t-distribution
        elif self.current_regime == "normal":
            extremity_factor *= 0.8  # More likely to use normal distribution

        # Calculate quantile using blended distribution
        if extremity_factor < 0.5:
            d2 = norm.ppf(P)
        else:
            df = max(2, 5 - 3 * extremity_factor)
            d2 = t.ppf(P, df)
            t_adjustment = 1.0 + 0.2 * (1.0 - df / 5) * d2**2 / df
            d2 = d2 * t_adjustment

        # Calculate base drift with current dynamic sigma
        base_mu = (
            d2 * self.sigma * np.sqrt(self.T) - np.log(S / K)
        ) / self.T + 0.5 * self.sigma**2

        # Apply enhanced psychology scaling with regime awareness
        if bullish_sentiment:
            psychology_factor = 1.0 + np.sin(np.pi / 2 * prob_distance) * self.psychology_bull_factor
        else:
            psychology_factor = 1.0 + np.sin(np.pi / 2 * abs_prob_distance) * self.psychology_bear_factor

        # Regime-based adjustment
        if self.current_regime == "trending":
            psychology_factor *= 1.2  # Amplify trends
        elif self.current_regime == "high_stress":
            psychology_factor *= 0.8  # Dampen extreme moves

        scaled_mu = base_mu * psychology_factor

        # Add non-linear component with regime awareness
        non_linear_component = np.sign(scaled_mu) * (abs_prob_distance**1.5) * 10
        if self.current_regime == "high_stress":
            non_linear_component *= 0.5  # Reduce non-linearity in stress

        final_mu = scaled_mu + non_linear_component * 0.2

        # Add anti-exploitation noise
        final_mu = self.add_anti_exploitation_noise(final_mu)

        return final_mu

    def generate_price_path(self, mu: float, duration_in_seconds: int, random_seed: Optional[int] = None) -> np.ndarray:
        """
        Generate price path with risk controls and anti-exploitation measures
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Calculate number of steps
        num_second_per_tick = int(1 / self.dt / (86_400 * 365))
        num_step = int(duration_in_seconds / num_second_per_tick)

        # Initialize arrays
        price_path = np.zeros(num_step + 1)
        price_path[0] = self.S_0
        increments = np.zeros(num_step)
        regime_state = np.zeros(num_step)

        # Generate correlated random increments with anti-exploitation noise
        for i in range(num_step):
            z = np.random.normal(0, 1)
            
            # Add anti-exploitation noise to random increments
            z = self.add_anti_exploitation_noise(z, self.noise_injection_level * 0.5)

            if i < self.memory_length:
                increments[i] = z
            else:
                memory_weights = np.power(
                    np.arange(1, self.memory_length + 1, 1), self.hurst - 1.5
                )
                memory_weights = memory_weights / np.sum(memory_weights)
                memory_effect = np.sum(
                    memory_weights * increments[i - self.memory_length : i]
                )
                increments[i] = 0.7 * z + 0.3 * memory_effect

        # Generate path with risk controls
        daily_movement = 0.0
        steps_per_day = int(86400 / num_second_per_tick)
        
        for i in range(num_step):
            current_price = price_path[i]

            # Update regime state
            if i > 0:
                if i > 10:
                    local_returns = np.diff(np.log(price_path[max(0, i - 10) : i + 1]))
                    local_vol = np.std(local_returns) / np.sqrt(self.dt)
                    target_regime = np.tanh((local_vol / self.sigma - 1) * 2)
                    regime_state[i] = 0.9 * regime_state[i - 1] + 0.1 * target_regime
                else:
                    regime_state[i] = regime_state[i - 1]

            # Calculate adaptive volatility with risk controls
            effective_sigma = min(self.max_volatility, self.sigma * (1 + 0.5 * regime_state[i]))

            # Mean reversion with adaptive strength
            log_price_ratio = np.log(current_price / self.S_0)
            base_reversion = -self.mean_reversion_strength * log_price_ratio
            adaptive_factor = 1.0 + 0.5 * np.tanh(abs(log_price_ratio) - 0.1)
            mean_reversion = base_reversion * adaptive_factor

            # Combine drift components
            effective_drift = mu + mean_reversion

            # Generate step increment
            drift_term = (effective_drift - 0.5 * effective_sigma**2) * self.dt
            diffusion_term = effective_sigma * np.sqrt(self.dt) * increments[i]
            log_return = drift_term + diffusion_term

            # Apply daily movement constraint
            if i % steps_per_day == 0:
                daily_movement = 0.0  # Reset daily movement counter
            
            proposed_return = log_return
            if abs(daily_movement + proposed_return) > self.max_daily_movement:
                # Scale down the return to stay within daily limits
                remaining_movement = self.max_daily_movement - abs(daily_movement)
                proposed_return = np.sign(proposed_return) * min(abs(proposed_return), remaining_movement)
            
            daily_movement += proposed_return

            # Update price
            price_path[i + 1] = current_price * np.exp(proposed_return)

        return price_path

    def calculate_optimization_metrics(self, index_values: np.ndarray, exposures: np.ndarray, 
                                     trades: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Calculate key metrics for parameter optimization
        
        Parameters:
        -----------
        index_values : np.ndarray
            Index price values
        exposures : np.ndarray
            Corresponding exposure values
        trades : pd.DataFrame, optional
            Trade data for additional metrics
            
        Returns:
        --------
        Dict[str, float]
            Optimization metrics
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
            # Higher entropy = less predictable = lower exploitability
            hist, _ = np.histogram(returns, bins=20)
            hist = hist / np.sum(hist)
            hist = hist[hist > 0]  # Remove zero probabilities
            entropy = -np.sum(hist * np.log(hist))
            max_entropy = np.log(len(hist))
            pattern_score = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
            
            exploitability = (autocorr + pattern_score) / 2
        else:
            exploitability = 0.5

        # 3. Risk Score (lower is better) - Focus on directional bias, not volatility
        if len(index_values) > 1:
            returns = np.diff(index_values) / index_values[:-1]
            
            # Directional bias risk (trending in one direction)
            # This is the real risk for brokers - sustained movement in one direction
            cumulative_return = np.sum(returns)
            directional_bias = abs(cumulative_return)  # Absolute cumulative return
            
            # Maximum drawdown (sustained losses)
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(np.min(drawdown))
            
            # Trend persistence (how long trends last - longer trends = higher risk)
            # Count consecutive moves in same direction
            signs = np.sign(returns)
            sign_changes = np.sum(np.diff(signs) != 0)
            trend_persistence = 1.0 - (sign_changes / max(1, len(returns) - 1))  # 0 = many reversals, 1 = persistent trend
            
            # Skewness risk (asymmetric returns favor one direction)
            if len(returns) > 3:
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    skewness = np.mean(((returns - mean_return) / std_return) ** 3)
                    skewness_risk = abs(skewness)  # High absolute skewness = directional bias
                else:
                    skewness_risk = 0.0
            else:
                skewness_risk = 0.0
            
            # Combined risk score (directional bias is the main risk, not volatility)
            risk_score = (
                directional_bias * 2.0 +      # Main risk: sustained directional movement
                max_drawdown * 1.5 +          # Drawdown risk
                trend_persistence * 1.0 +     # Trend persistence risk
                skewness_risk * 0.5           # Asymmetry risk
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

    def optimize_parameters_bayesian(self, historical_data: Dict, n_calls: int = 50) -> Dict[str, float]:
        """
        Optimize parameters using Bayesian optimization
        
        Parameters:
        -----------
        historical_data : Dict
            Historical data containing exposures, prices, trades
        n_calls : int
            Number of optimization iterations
            
        Returns:
        --------
        Dict[str, float]
            Optimized parameters
        """
        if not BAYESIAN_OPT_AVAILABLE:
            print("Bayesian optimization not available. Using scipy optimization.")
            return self.optimize_parameters_scipy(historical_data)

        # Define parameter search space
        space = [
            Real(0.2, 0.8, name='sigma'),
            Real(80000, 200000, name='scale'),
            Real(0.3, 0.5, name='k'),
            Real(1.0, 3.0, name='smoothness_factor'),
            Real(0.2, 0.6, name='psychology_bull_factor'),
            Real(0.3, 0.7, name='psychology_bear_factor'),
            Real(0.01, 0.05, name='noise_injection_level')
        ]

        def objective(params):
            """Objective function for optimization"""
            sigma, scale, k, smoothness, psych_bull, psych_bear, noise_level = params
            
            # Temporarily set parameters
            original_params = {
                'sigma_base': self.sigma_base,
                'scale_base': self.scale_base,
                'k_base': self.k_base,
                'smoothness_factor': self.smoothness_factor,
                'psychology_bull_factor': self.psychology_bull_factor,
                'psychology_bear_factor': self.psychology_bear_factor,
                'noise_injection_level': self.noise_injection_level
            }
            
            self.sigma_base = sigma
            self.scale_base = scale
            self.k_base = k
            self.smoothness_factor = smoothness
            self.psychology_bull_factor = psych_bull
            self.psychology_bear_factor = psych_bear
            self.noise_injection_level = noise_level
            
            try:
                # Run simulation with these parameters
                metrics = self.run_simulation_with_params(historical_data)
                
                # Calculate weighted objective (minimize)
                objective_value = (
                    -0.3 * metrics['responsiveness'] +      # Want high responsiveness
                    0.4 * metrics['exploitability'] +       # Want low exploitability
                    0.2 * metrics['risk'] +                 # Want low risk
                    -0.1 * metrics['unpredictability']      # Want high unpredictability
                )
                
            except Exception as e:
                print(f"Error in objective function: {e}")
                objective_value = 1.0  # High penalty for failed simulations
            
            # Restore original parameters
            for key, value in original_params.items():
                setattr(self, key, value)
            
            return objective_value

        # Run optimization
        print(f"Starting Bayesian optimization with {n_calls} iterations...")
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            random_state=42,
            acq_func='EI'  # Expected Improvement
        )

        # Extract optimal parameters
        optimal_params = {
            'sigma': result.x[0],
            'scale': result.x[1],
            'k': result.x[2],
            'smoothness_factor': result.x[3],
            'psychology_bull_factor': result.x[4],
            'psychology_bear_factor': result.x[5],
            'noise_injection_level': result.x[6]
        }

        print(f"Optimization completed. Best objective value: {result.fun:.4f}")
        
        # Store optimization history
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'method': 'bayesian',
            'optimal_params': optimal_params,
            'objective_value': result.fun,
            'n_iterations': n_calls
        })

        return optimal_params

    def optimize_parameters_scipy(self, historical_data: Dict) -> Dict[str, float]:
        """
        Optimize parameters using scipy optimization (fallback)
        
        Parameters:
        -----------
        historical_data : Dict
            Historical data containing exposures, prices, trades
            
        Returns:
        --------
        Dict[str, float]
            Optimized parameters
        """
        # Define bounds for parameters
        bounds = [
            (0.2, 0.8),    # sigma
            (80000, 200000),  # scale
            (0.3, 0.5),    # k
            (1.0, 3.0),    # smoothness_factor
            (0.2, 0.6),    # psychology_bull_factor
            (0.3, 0.7),    # psychology_bear_factor
            (0.01, 0.05)   # noise_injection_level
        ]

        def objective(params):
            """Objective function for scipy optimization"""
            sigma, scale, k, smoothness, psych_bull, psych_bear, noise_level = params
            
            # Store original parameters
            original_params = {
                'sigma_base': self.sigma_base,
                'scale_base': self.scale_base,
                'k_base': self.k_base,
                'smoothness_factor': self.smoothness_factor,
                'psychology_bull_factor': self.psychology_bull_factor,
                'psychology_bear_factor': self.psychology_bear_factor,
                'noise_injection_level': self.noise_injection_level
            }
            
            # Set new parameters
            self.sigma_base = sigma
            self.scale_base = scale
            self.k_base = k
            self.smoothness_factor = smoothness
            self.psychology_bull_factor = psych_bull
            self.psychology_bear_factor = psych_bear
            self.noise_injection_level = noise_level
            
            try:
                # Run simulation with these parameters
                metrics = self.run_simulation_with_params(historical_data)
                
                # Calculate weighted objective (minimize)
                objective_value = (
                    -0.3 * metrics['responsiveness'] +      # Want high responsiveness
                    0.4 * metrics['exploitability'] +       # Want low exploitability
                    0.2 * metrics['risk'] +                 # Want low risk
                    -0.1 * metrics['unpredictability']      # Want high unpredictability
                )
                
            except Exception as e:
                print(f"Error in objective function: {e}")
                objective_value = 1.0  # High penalty for failed simulations
            
            # Restore original parameters
            for key, value in original_params.items():
                setattr(self, key, value)
            
            return objective_value

        # Initial guess
        x0 = [
            self.sigma_base,
            self.scale_base,
            self.k_base,
            self.smoothness_factor,
            self.psychology_bull_factor,
            self.psychology_bear_factor,
            self.noise_injection_level
        ]

        print("Starting scipy optimization...")
        result = minimize(
            fun=objective,
            x0=x0,
            bounds=bounds,
            method='L-BFGS-B'
        )

        # Extract optimal parameters
        optimal_params = {
            'sigma': result.x[0],
            'scale': result.x[1],
            'k': result.x[2],
            'smoothness_factor': result.x[3],
            'psychology_bull_factor': result.x[4],
            'psychology_bear_factor': result.x[5],
            'noise_injection_level': result.x[6]
        }

        print(f"Optimization completed. Best objective value: {result.fun:.4f}")
        
        # Store optimization history
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'method': 'scipy',
            'optimal_params': optimal_params,
            'objective_value': result.fun,
            'success': result.success
        })

        return optimal_params

    def run_simulation_with_params(self, historical_data: Dict) -> Dict[str, float]:
        """
        Run simulation with current parameters and return metrics
        
        Parameters:
        -----------
        historical_data : Dict
            Historical data for simulation
            
        Returns:
        --------
        Dict[str, float]
            Simulation metrics
        """
        # Extract data
        exposures = historical_data.get('exposures', [])
        if len(exposures) == 0:
            return {'responsiveness': 0, 'exploitability': 1, 'risk': 1, 'unpredictability': 0}
        
        # Generate index values
        index_values = [self.S_0]
        
        for exposure in exposures:
            # Map exposure to probability
            prob = self.exposure_to_probability(exposure)
            
            # Map probability to drift
            mu = self.compute_mu_from_probability(prob)
            
            # Generate short price path (simplified for optimization)
            path = self.generate_price_path(mu, 300, random_seed=42)  # 5 minutes
            
            # Take final price
            index_values.append(path[-1])
        
        # Calculate metrics
        index_values = np.array(index_values)
        exposures = np.array(exposures)
        
        return self.calculate_optimization_metrics(index_values, exposures)

    def apply_risk_constraints(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Apply hard constraints to ensure broker safety
        
        Parameters:
        -----------
        params : Dict[str, float]
            Parameters to constrain
            
        Returns:
        --------
        Dict[str, float]
            Constrained parameters
        """
        constrained_params = params.copy()
        
        # Apply volatility constraint
        if constrained_params.get('sigma', 0) > self.max_volatility:
            constrained_params['sigma'] = self.max_volatility
        
        # Apply minimum smoothing constraint
        if constrained_params.get('noise_injection_level', 0) < self.min_smoothing:
            constrained_params['noise_injection_level'] = self.min_smoothing
        
        # Ensure scale prevents over-sensitivity
        min_scale = self.calculate_min_scale_for_sensitivity()
        if constrained_params.get('scale', 0) < min_scale:
            constrained_params['scale'] = min_scale
        
        return constrained_params

    def calculate_min_scale_for_sensitivity(self) -> float:
        """
        Calculate minimum scale to prevent over-sensitivity to single trades
        
        Returns:
        --------
        float
            Minimum scale value
        """
        # Assume maximum single trade size and calculate required scale
        max_trade_size = 100000  # $100k max trade
        min_scale = max_trade_size / self.max_sensitivity
        return min_scale

    def daily_risk_check(self, index_performance: np.ndarray) -> Dict[str, Any]:
        """
        Daily risk assessment and parameter adjustment
        
        Parameters:
        -----------
        index_performance : np.ndarray
            Recent index performance data
            
        Returns:
        --------
        Dict[str, Any]
            Risk assessment results and adjustments made
        """
        if len(index_performance) < 2:
            return {'status': 'insufficient_data'}
        
        # Calculate daily return
        daily_return = (index_performance[-1] / index_performance[-2]) - 1
        
        # Calculate rolling volatility
        if len(index_performance) >= 30:
            returns = np.diff(index_performance) / index_performance[:-1]
            volatility = np.std(returns[-30:]) * np.sqrt(252)  # 30-day annualized vol
        else:
            volatility = 0.0
        
        adjustments_made = []
        
        # Risk triggers
        if abs(daily_return) > self.max_daily_movement:
            # Increase smoothing, reduce sensitivity
            self.noise_injection_level = min(0.05, self.noise_injection_level * 1.2)
            self.scale_base = self.scale_base * 1.1  # Reduce sensitivity
            adjustments_made.append('increased_smoothing_reduced_sensitivity')
        
        if volatility > 0.6:  # 60% annualized volatility
            # Reduce sigma, increase scale
            self.sigma_base = max(0.2, self.sigma_base * 0.9)
            self.scale_base = self.scale_base * 1.1
            adjustments_made.append('reduced_volatility')
        
        # Store risk metrics
        risk_metrics = {
            'timestamp': datetime.now(),
            'daily_return': daily_return,
            'volatility': volatility,
            'adjustments_made': adjustments_made,
            'current_params': {
                'sigma_base': self.sigma_base,
                'scale_base': self.scale_base,
                'noise_injection_level': self.noise_injection_level
            }
        }
        
        self.risk_metrics_history.append(risk_metrics)
        
        return risk_metrics

    def run_comprehensive_backtest(self, historical_data: Dict, 
                                 optimization_method: str = 'bayesian',
                                 n_optimization_calls: int = 50) -> Dict[str, Any]:
        """
        Run comprehensive backtesting with parameter optimization
        
        Parameters:
        -----------
        historical_data : Dict
            Historical data for backtesting
        optimization_method : str
            'bayesian' or 'scipy'
        n_optimization_calls : int
            Number of optimization iterations
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive backtest results
        """
        print("Starting comprehensive backtesting...")
        
        # Phase 1: Baseline performance
        print("Phase 1: Baseline performance evaluation...")
        baseline_metrics = self.run_simulation_with_params(historical_data)
        
        # Phase 2: Parameter optimization
        print("Phase 2: Parameter optimization...")
        if optimization_method == 'bayesian':
            optimal_params = self.optimize_parameters_bayesian(historical_data, n_optimization_calls)
        else:
            optimal_params = self.optimize_parameters_scipy(historical_data)
        
        # Apply risk constraints
        optimal_params = self.apply_risk_constraints(optimal_params)
        
        # Phase 3: Apply optimal parameters and re-test
        print("Phase 3: Testing optimized parameters...")
        
        # Store original parameters
        original_params = {
            'sigma_base': self.sigma_base,
            'scale_base': self.scale_base,
            'k_base': self.k_base,
            'smoothness_factor': self.smoothness_factor,
            'psychology_bull_factor': self.psychology_bull_factor,
            'psychology_bear_factor': self.psychology_bear_factor,
            'noise_injection_level': self.noise_injection_level
        }
        
        # Apply optimal parameters
        for key, value in optimal_params.items():
            if hasattr(self, f"{key}_base"):
                setattr(self, f"{key}_base", value)
            else:
                setattr(self, key, value)
        
        # Test optimized performance
        optimized_metrics = self.run_simulation_with_params(historical_data)
        
        # Phase 4: Exploitation testing
        print("Phase 4: Anti-exploitation testing...")
        exploitation_results = self.test_exploitation_resistance(historical_data)
        
        # Compile results
        backtest_results = {
            'baseline_metrics': baseline_metrics,
            'optimal_params': optimal_params,
            'optimized_metrics': optimized_metrics,
            'exploitation_results': exploitation_results,
            'improvement': {
                'responsiveness': optimized_metrics['responsiveness'] - baseline_metrics['responsiveness'],
                'exploitability': baseline_metrics['exploitability'] - optimized_metrics['exploitability'],  # Lower is better
                'risk': baseline_metrics['risk'] - optimized_metrics['risk'],  # Lower is better
                'unpredictability': optimized_metrics['unpredictability'] - baseline_metrics['unpredictability']
            },
            'optimization_history': self.optimization_history,
            'timestamp': datetime.now()
        }
        
        # Restore original parameters
        for key, value in original_params.items():
            setattr(self, key, value)
        
        print("Comprehensive backtesting completed!")
        return backtest_results

    def test_exploitation_resistance(self, historical_data: Dict) -> Dict[str, Any]:
        """
        Test resistance to various exploitation strategies
        
        Parameters:
        -----------
        historical_data : Dict
            Historical data for testing
            
        Returns:
        --------
        Dict[str, Any]
            Exploitation resistance test results
        """
        print("Testing exploitation resistance...")
        
        exploitation_tests = {}
        
        # Test 1: Pattern prediction
        try:
            exposures = historical_data.get('exposures', [])
            if len(exposures) > 20:
                # Generate index values
                index_values = []
                for exposure in exposures[:20]:
                    prob = self.exposure_to_probability(exposure)
                    mu = self.compute_mu_from_probability(prob)
                    path = self.generate_price_path(mu, 300, random_seed=42)
                    index_values.append(path[-1])
                
                # Test autocorrelation
                returns = np.diff(np.log(index_values))
                autocorr = abs(np.corrcoef(returns[:-1], returns[1:])[0, 1])
                exploitation_tests['pattern_prediction'] = {
                    'autocorrelation': autocorr,
                    'resistance_score': 1.0 - autocorr,  # Higher is better
                    'status': 'low_risk' if autocorr < 0.1 else 'medium_risk' if autocorr < 0.3 else 'high_risk'
                }
        except Exception as e:
            exploitation_tests['pattern_prediction'] = {'error': str(e)}
        
        # Test 2: Boundary exploitation
        try:
            # Test extreme exposures
            extreme_exposures = [-1000000, -500000, 0, 500000, 1000000]
            probabilities = [self.exposure_to_probability(exp) for exp in extreme_exposures]
            
            # Check if probabilities are properly bounded
            min_prob, max_prob = min(probabilities), max(probabilities)
            boundary_exploitation = max(abs(min_prob - (0.5 - self.k)), abs(max_prob - (0.5 + self.k)))
            
            exploitation_tests['boundary_exploitation'] = {
                'boundary_violation': boundary_exploitation,
                'resistance_score': 1.0 - boundary_exploitation,
                'status': 'low_risk' if boundary_exploitation < 0.01 else 'medium_risk' if boundary_exploitation < 0.05 else 'high_risk'
            }
        except Exception as e:
            exploitation_tests['boundary_exploitation'] = {'error': str(e)}
        
        # Test 3: Timing exploitation
        try:
            # Test if same exposure gives different results at different times
            test_exposure = 50000
            results = []
            for i in range(10):
                # Simulate different times
                self.update_dynamic_parameters(datetime.now() + timedelta(hours=i))
                prob = self.exposure_to_probability(test_exposure)
                results.append(prob)
            
            timing_variation = np.std(results)
            exploitation_tests['timing_exploitation'] = {
                'timing_variation': timing_variation,
                'resistance_score': min(1.0, timing_variation * 10),  # Some variation is good
                'status': 'good' if 0.001 < timing_variation < 0.01 else 'needs_adjustment'
            }
        except Exception as e:
            exploitation_tests['timing_exploitation'] = {'error': str(e)}
        
        return exploitation_tests

    def generate_optimization_report(self, backtest_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive optimization report
        
        Parameters:
        -----------
        backtest_results : Dict[str, Any]
            Results from comprehensive backtesting
            
        Returns:
        --------
        str
            Formatted optimization report
        """
        report = []
        report.append("=" * 80)
        report.append("RISK-OPTIMIZED SUPPLY-DEMAND INDEX - OPTIMIZATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Baseline vs Optimized Performance
        baseline = backtest_results['baseline_metrics']
        optimized = backtest_results['optimized_metrics']
        improvement = backtest_results['improvement']
        
        report.append("PERFORMANCE COMPARISON")
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
                report.append(f"{param:<25}: {value:.4f}")
            else:
                report.append(f"{param:<25}: {value}")
        report.append("")
        
        # Exploitation Resistance
        exploitation = backtest_results['exploitation_results']
        report.append("EXPLOITATION RESISTANCE TESTS")
        report.append("-" * 40)
        for test_name, results in exploitation.items():
            if 'error' not in results:
                status = results.get('status', 'unknown')
                score = results.get('resistance_score', 0)
                report.append(f"{test_name:<25}: {status:<15} (Score: {score:.3f})")
            else:
                report.append(f"{test_name:<25}: ERROR - {results['error']}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        if improvement['responsiveness'] > 0.1:
            report.append("✓ Significant improvement in responsiveness achieved")
        elif improvement['responsiveness'] < -0.05:
            report.append("⚠ Responsiveness decreased - consider adjusting parameters")
        
        if improvement['exploitability'] > 0.1:
            report.append("✓ Significant reduction in exploitability achieved")
        elif improvement['exploitability'] < -0.05:
            report.append("⚠ Exploitability increased - review anti-exploitation measures")
        
        if improvement['risk'] > 0.1:
            report.append("✓ Significant risk reduction achieved")
        elif improvement['risk'] < -0.05:
            report.append("⚠ Risk increased - review risk constraints")
        
        # Overall assessment
        overall_score = (
            improvement['responsiveness'] * 0.3 +
            improvement['exploitability'] * 0.4 +
            improvement['risk'] * 0.2 +
            improvement['unpredictability'] * 0.1
        )
        
        report.append("")
        report.append(f"OVERALL OPTIMIZATION SCORE: {overall_score:.3f}")
        if overall_score > 0.1:
            report.append("✓ OPTIMIZATION SUCCESSFUL - Parameters significantly improved")
        elif overall_score > 0:
            report.append("~ OPTIMIZATION MODERATE - Some improvements achieved")
        else:
            report.append("⚠ OPTIMIZATION INEFFECTIVE - Consider different approach")
        
        report.append("=" * 80)
        
        return "\n".join(report)

    def process_exposure_data(
        self,
        exposure_data: List[float],
        duration_in_seconds: int = 3600,
        num_paths_per_exposure: int = 100,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Process exposure data and generate comprehensive results
        
        Parameters:
        -----------
        exposure_data : List[float]
            List of exposure values to process
        duration_in_seconds : int
            Duration for each price path simulation
        num_paths_per_exposure : int
            Number of paths to generate per exposure
        random_seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        Dict[str, Any]
            Comprehensive results including paths, probabilities, and metrics
        """
        np.random.seed(random_seed)
        
        results = {
            'exposures': exposure_data,
            'probabilities': [],
            'drifts': [],
            'price_paths': [],
            'final_prices': [],
            'metrics': []
        }
        
        for i, exposure in enumerate(exposure_data):
            # Map exposure to probability
            probability = self.exposure_to_probability(exposure)
            results['probabilities'].append(probability)
            
            # Map probability to drift
            mu = self.compute_mu_from_probability(probability)
            results['drifts'].append(mu)
            
            # Generate multiple price paths for this exposure
            paths_for_exposure = []
            final_prices_for_exposure = []
            
            for j in range(num_paths_per_exposure):
                path = self.generate_price_path(
                    mu=mu,
                    duration_in_seconds=duration_in_seconds,
                    random_seed=random_seed + i * 1000 + j
                )
                paths_for_exposure.append(path)
                final_prices_for_exposure.append(path[-1])
            
            results['price_paths'].append(paths_for_exposure)
            results['final_prices'].append(final_prices_for_exposure)
            
            # Calculate metrics for this exposure
            metrics = {
                'mean_final_price': np.mean(final_prices_for_exposure),
                'std_final_price': np.std(final_prices_for_exposure),
                'min_final_price': np.min(final_prices_for_exposure),
                'max_final_price': np.max(final_prices_for_exposure),
                'probability': probability,
                'drift': mu,
                'exposure': exposure
            }
            results['metrics'].append(metrics)
        
        return results

    def create_comprehensive_visualization(
        self,
        results: Dict[str, Any],
        max_paths_to_plot: int = 30,
        figsize: Tuple[int, int] = (15, 10),
        save_path: str = None
    ) -> None:
        """
        Create comprehensive visualization split into multiple clear plots
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Results from process_exposure_data
        max_paths_to_plot : int
            Maximum number of paths to plot per exposure
        figsize : Tuple[int, int]
            Figure size for each plot
        save_path : str, optional
            Base path to save the plots (will add suffixes)
        """
        exposures = results['exposures']
        probabilities = results['probabilities']
        drifts = results['drifts']
        price_paths = results['price_paths']
        metrics = results['metrics']
        
        # Determine save directory
        if save_path:
            save_dir = save_path.rsplit('/', 1)[0] if '/' in save_path else '.'
            base_name = save_path.rsplit('/', 1)[-1].rsplit('.', 1)[0] if '/' in save_path else save_path.rsplit('.', 1)[0]
        else:
            save_dir = 'plots/risk'
            base_name = 'analysis'
        
        # Plot 1: Exposure-Probability-Drift Relationships
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Risk-Optimized Index: Exposure-Response Mapping', fontsize=16, fontweight='bold')
        
        # Exposure to Probability
        ax = axes[0, 0]
        scatter = ax.scatter(exposures, probabilities, alpha=0.7, c=exposures, cmap='RdYlBu_r', s=50)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Neutral (50%)')
        ax.axhline(y=0.5 + self.k_base, color='orange', linestyle=':', alpha=0.8, label=f'Upper Bound ({0.5 + self.k_base:.1%})')
        ax.axhline(y=0.5 - self.k_base, color='orange', linestyle=':', alpha=0.8, label=f'Lower Bound ({0.5 - self.k_base:.1%})')
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
        ax.axhline(y=self.S_0, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Starting Price ({self.S_0:,.0f})')
        ax.set_xlabel('Net Exposure ($)', fontsize=12)
        ax.set_ylabel('Mean Final Price', fontsize=12)
        ax.set_title('Exposure → Expected Price Impact', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.colorbar(scatter3, ax=ax, label='Final Price')
        
        # Price Volatility vs Exposure
        ax = axes[1, 1]
        price_volatilities = [m['std_final_price'] for m in metrics]
        scatter4 = ax.scatter(exposures, price_volatilities, alpha=0.7, c=price_volatilities, cmap='Reds', s=50)
        ax.set_xlabel('Net Exposure ($)', fontsize=12)
        ax.set_ylabel('Price Volatility (Std Dev)', fontsize=12)
        ax.set_title('Exposure → Price Uncertainty', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter4, ax=ax, label='Volatility')
        
        plt.tight_layout()
        if save_path:
            plot1_path = f"{save_dir}/{base_name}_1_exposure_response.png"
            plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
            print(f"Exposure-Response plot saved to {plot1_path}")
        plt.show()
        
        # Plot 2: Price Path Analysis
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Risk-Optimized Index: Price Path Analysis', fontsize=16, fontweight='bold')
        
        # Sample Price Paths
        ax = axes[0, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, min(8, len(exposures))))
        
        for i in range(min(8, len(exposures))):
            paths = price_paths[i]
            exposure = exposures[i]
            
            # Plot representative paths
            for j, path in enumerate(paths[:min(5, max_paths_to_plot)]):
                alpha = 0.6 if j == 0 else 0.3
                linewidth = 2 if j == 0 else 1
                label = f'Exp: {exposure:,.0f}' if j == 0 else None
                ax.plot(path, color=colors[i], alpha=alpha, linewidth=linewidth, label=label)
        
        ax.axhline(y=self.S_0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Starting Price')
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Index Price', fontsize=12)
        ax.set_title('Sample Price Paths by Exposure', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Final Price Distribution
        ax = axes[0, 1]
        all_final_prices = []
        for final_prices in results['final_prices']:
            all_final_prices.extend(final_prices)
        
        ax.hist(all_final_prices, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(x=self.S_0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Starting Price')
        ax.set_xlabel('Final Price', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Final Price Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Risk Metrics
        ax = axes[1, 0]
        price_ranges = [m['max_final_price'] - m['min_final_price'] for m in metrics]
        scatter5 = ax.scatter(exposures, price_ranges, alpha=0.7, c=price_ranges, cmap='Oranges', s=50)
        ax.set_xlabel('Net Exposure ($)', fontsize=12)
        ax.set_ylabel('Price Range (Max - Min)', fontsize=12)
        ax.set_title('Exposure → Price Risk Range', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter5, ax=ax, label='Price Range')
        
        # Parameter Summary
        ax = axes[1, 1]
        ax.text(0.05, 0.95, 'Engine Configuration', fontsize=14, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, 0.85, f'Base Parameters:', fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, 0.78, f'• σ (Volatility): {self.sigma_base:.1%}', fontsize=10, transform=ax.transAxes)
        ax.text(0.05, 0.71, f'• Scale: {self.scale_base:,.0f}', fontsize=10, transform=ax.transAxes)
        ax.text(0.05, 0.64, f'• k (Range): ±{self.k_base:.1%}', fontsize=10, transform=ax.transAxes)
        ax.text(0.05, 0.57, f'• S₀: {self.S_0:,.0f}', fontsize=10, transform=ax.transAxes)
        
        ax.text(0.05, 0.45, f'Risk Controls:', fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, 0.38, f'• Max Daily Move: {self.max_daily_movement:.1%}', fontsize=10, transform=ax.transAxes)
        ax.text(0.05, 0.31, f'• Noise Level: {self.noise_injection_level:.1%}', fontsize=10, transform=ax.transAxes)
        ax.text(0.05, 0.24, f'• Current Regime: {self.current_regime}', fontsize=10, transform=ax.transAxes)
        
        ax.text(0.05, 0.12, f'Anti-Exploitation:', fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, 0.05, f'• Pattern Breaking: {self.pattern_breaking_frequency:.1%}', fontsize=10, transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plot2_path = f"{save_dir}/{base_name}_2_price_analysis.png"
            plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
            print(f"Price Analysis plot saved to {plot2_path}")
        plt.show()
        
        # Plot 3: Risk and Performance Metrics
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Risk-Optimized Index: Performance & Risk Metrics', fontsize=16, fontweight='bold')
        
        # Responsiveness Analysis
        ax = axes[0, 0]
        correlations = []
        for i, (exp, prob) in enumerate(zip(exposures, probabilities)):
            # Calculate local responsiveness
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
        
        # Exploitation Risk Heatmap
        ax = axes[0, 1]
        # Create a simple risk matrix
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
        volatilities = [m['std_final_price'] / m['mean_final_price'] for m in metrics if m['mean_final_price'] > 0]
        ax.hist(volatilities, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('Coefficient of Variation', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Price Volatility Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Performance Summary
        ax = axes[1, 1]
        # Calculate key performance metrics
        total_exposures = len(exposures)
        avg_prob = np.mean(probabilities)
        prob_range = max(probabilities) - min(probabilities)
        avg_volatility = np.mean([m['std_final_price'] for m in metrics])
        
        ax.text(0.05, 0.9, 'Performance Summary', fontsize=14, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, 0.8, f'Total Scenarios: {total_exposures}', fontsize=11, transform=ax.transAxes)
        ax.text(0.05, 0.73, f'Avg Probability: {avg_prob:.3f}', fontsize=11, transform=ax.transAxes)
        ax.text(0.05, 0.66, f'Probability Range: {prob_range:.3f}', fontsize=11, transform=ax.transAxes)
        ax.text(0.05, 0.59, f'Avg Price Volatility: {avg_volatility:,.0f}', fontsize=11, transform=ax.transAxes)
        
        # Risk indicators
        high_risk_scenarios = sum(1 for m in metrics if abs(m['exposure']) > np.std(exposures) * 2)
        ax.text(0.05, 0.45, 'Risk Indicators:', fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, 0.38, f'High Risk Scenarios: {high_risk_scenarios}', fontsize=11, transform=ax.transAxes)
        ax.text(0.05, 0.31, f'Risk Ratio: {high_risk_scenarios/total_exposures:.1%}', fontsize=11, transform=ax.transAxes)
        
        # Anti-exploitation status
        ax.text(0.05, 0.17, 'Anti-Exploitation:', fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.05, 0.1, f'Noise Injection: Active', fontsize=11, transform=ax.transAxes)
        ax.text(0.05, 0.03, f'Pattern Breaking: {self.pattern_breaking_frequency:.1%}', fontsize=11, transform=ax.transAxes)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plot3_path = f"{save_dir}/{base_name}_3_risk_metrics.png"
            plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
            print(f"Risk Metrics plot saved to {plot3_path}")
        plt.show()

    def generate_parameter_classification_report(self) -> str:
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
        report.append("Risk-Optimized Supply-Demand Index Engine")
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
            ("S_0", self.S_0, "Starting price reference point", "Critical for price calculations"),
            ("T", self.T, "Time horizon for calculations", "Affects all drift computations"),
            ("dt", self.dt, "Time step size", "Core simulation parameter"),
            ("memory_length", self.memory_length, "Fractional process memory", "Affects path generation"),
            ("fractal_dimension", self.fractal_dimension, "Market microstructure", "Controls price path characteristics"),
            ("hurst", self.hurst, "Derived from fractal dimension", "Auto-calculated parameter")
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
            ("noise_injection_level", self.noise_injection_level, "Anti-exploitation noise", 
             "Increase if exploitation detected, decrease if too much randomness"),
            ("pattern_breaking_frequency", self.pattern_breaking_frequency, "Pattern disruption rate",
             "Increase during high-frequency trading periods"),
            ("scale", self.scale_base, "Exposure sensitivity", 
             "Increase during high volatility, decrease during calm periods"),
            ("k", self.k_base, "Probability range", 
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
            ("sigma_base", self.sigma_base, "Base volatility", 
             "Adjust based on market volatility regime"),
            ("psychology_bull_factor", self.psychology_bull_factor, "Bullish psychology scaling",
             "Increase during bull markets, decrease during bear markets"),
            ("psychology_bear_factor", self.psychology_bear_factor, "Bearish psychology scaling",
             "Increase during bear markets, decrease during bull markets"),
            ("smoothness_factor", self.smoothness_factor, "Transition smoothness",
             "Increase for smoother transitions, decrease for more responsive behavior")
        ]
        
        for param, value, description, guidance in medium_priority:
            if isinstance(value, float):
                report.append(f"• {param:<25}: {value:.1%} ({description})")
            else:
                report.append(f"• {param:<25}: {value} ({description})")
            report.append(f"  └─ Guidance: {guidance}")
            report.append("")
        
        # Low Priority (Adjust monthly or during major events)
        report.append("LOW PRIORITY - Adjust Monthly or During Major Events")
        report.append("~" * 50)
        
        low_priority = [
            ("mean_reversion_strength", self.mean_reversion_strength, "Price mean reversion",
             "Adjust based on long-term market trends"),
            ("sigma_noise_std", self.sigma_noise_std, "Volatility noise variation",
             "Adjust based on volatility of volatility"),
            ("scale_adaptation_rate", self.scale_adaptation_rate, "Scale adaptation speed",
             "Adjust based on how quickly market conditions change"),
            ("k_oscillation_amplitude", self.k_oscillation_amplitude, "K parameter oscillation",
             "Adjust based on exploitation attempts frequency")
        ]
        
        for param, value, description, guidance in low_priority:
            if isinstance(value, float):
                report.append(f"• {param:<25}: {value:.1%} ({description})")
            else:
                report.append(f"• {param:<25}: {value} ({description})")
            report.append(f"  └─ Guidance: {guidance}")
            report.append("")
        
        # Risk Constraints (Emergency adjustments only)
        report.append("RISK CONSTRAINTS - Emergency Adjustments Only")
        report.append("-" * 50)
        report.append("These parameters should only be changed in emergency situations")
        report.append("or when fundamental risk tolerance changes.")
        report.append("")
        
        risk_constraints = [
            ("max_daily_movement", self.max_daily_movement, "Maximum daily price movement",
             "Only increase if business can handle more risk"),
            ("max_volatility", self.max_volatility, "Maximum allowed volatility",
             "Only increase during extreme market conditions"),
            ("min_smoothing", self.min_smoothing, "Minimum smoothing level",
             "Only decrease if anti-exploitation is too strong"),
            ("max_sensitivity", self.max_sensitivity, "Maximum sensitivity to single trade",
             "Only increase if confident about trade size limits")
        ]
        
        for param, value, description, guidance in risk_constraints:
            if isinstance(value, float):
                report.append(f"• {param:<25}: {value:.1%} ({description})")
            else:
                report.append(f"• {param:<25}: {value} ({description})")
            report.append(f"  └─ WARNING: {guidance}")
            report.append("")
        
        # Adjustment Triggers
        report.append("ADJUSTMENT TRIGGERS")
        report.append("-" * 30)
        report.append("Conditions that should trigger parameter adjustments:")
        report.append("")
        
        triggers = [
            ("High Exploitation Risk", "Increase noise_injection_level, pattern_breaking_frequency"),
            ("Low Responsiveness", "Decrease scale, increase k, reduce smoothness_factor"),
            ("High Volatility Period", "Increase sigma_base, reduce scale sensitivity"),
            ("Trending Market", "Adjust psychology factors, modify k oscillation"),
            ("Risk Limit Breach", "Emergency adjustment of risk constraints"),
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
            "• Exploitability Score (target: < 0.3)",
            "• Risk Score (target: < 0.4)",
            "• Unpredictability Score (target: > 0.8)",
            "• Daily price movement vs max_daily_movement",
            "• Autocorrelation in returns (target: < 0.1)",
            "• Pattern detection attempts",
            "• Boundary exploitation attempts",
            "• Market regime changes"
        ]
        
        for item in monitoring:
            report.append(item)
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def run_risk_optimization_example():
    """
    Example usage of the Risk-Optimized Supply-Demand Index Engine
    """
    print("Running Risk-Optimized Supply-Demand Index Example...")

    # Initialize engine with risk-optimized parameters
    engine = RiskOptimizedSupplyDemandIndexEngine(
        sigma=0.35,  # 35% base volatility
        scale=140_000,  # Balanced exposure sensitivity
        k=0.38,  # Conservative probability range
        T=1 / (365 * 24),  # 1 hour time horizon
        S_0=100_000,  # Starting price
        mean_reversion_strength=0.05,  # Moderate mean reversion
        fractal_dimension=1.5,  # Standard Brownian motion
        memory_length=20,  # Memory length for fractional process
        smoothness_factor=2.0,  # Controls smoothness of transitions
        # Risk optimization parameters
        sigma_noise_std=0.05,  # 5% daily sigma variation
        scale_adaptation_rate=0.1,  # 10% adaptation rate
        k_oscillation_amplitude=0.05,  # 5% k oscillation
        noise_injection_level=0.015,  # 1.5% noise injection
        pattern_breaking_frequency=0.1,  # 10% pattern breaking
        psychology_bull_factor=0.35,  # Bullish psychology factor
        psychology_bear_factor=0.45,  # Bearish psychology factor
        # Risk constraints
        max_daily_movement=0.05,  # 5% max daily movement
        max_volatility=0.8,  # 80% max volatility
        min_smoothing=0.05,  # 5% minimum smoothing
        max_sensitivity=0.001,  # 0.1% max sensitivity to single trade
    )

    # Create sample historical data for optimization
    np.random.seed(42)
    sample_exposures = np.random.normal(0, 50000, 100)  # 100 random exposures
    historical_data = {
        'exposures': sample_exposures.tolist(),
        'timestamps': [datetime.now() + timedelta(minutes=i*5) for i in range(100)]
    }

    print(f"Created sample historical data with {len(sample_exposures)} exposure points")
    print(f"Exposure range: {min(sample_exposures):,.0f} to {max(sample_exposures):,.0f}")

    # Run comprehensive backtesting with optimization
    print("\nRunning comprehensive backtesting with parameter optimization...")
    backtest_results = engine.run_comprehensive_backtest(
        historical_data=historical_data,
        optimization_method='scipy',  # Use scipy as fallback
        n_optimization_calls=20  # Reduced for example
    )

    # Generate and print optimization report
    print("\nGenerating optimization report...")
    report = engine.generate_optimization_report(backtest_results)
    print(report)

    # Save results to file
    with open('risk_optimization_results.json', 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        serializable_results = {}
        for key, value in backtest_results.items():
            if key == 'timestamp':
                serializable_results[key] = value.isoformat()
            else:
                serializable_results[key] = value
        
        json.dump(serializable_results, f, indent=2, default=str)

    print("\nRisk optimization example completed!")
    print("Results saved to 'risk_optimization_results.json'")
    
    return engine, backtest_results


if __name__ == "__main__":
    # Run example analysis
    engine, results = run_risk_optimization_example()
