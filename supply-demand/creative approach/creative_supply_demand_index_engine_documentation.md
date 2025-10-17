# Creative Supply-Demand Index Engine Documentation

## Overview

The Creative Supply-Demand Index Engine represents the most mathematically sophisticated approach to supply-demand modeling, incorporating advanced stochastic processes, fractal characteristics, and behavioral finance principles. This engine pushes the boundaries of financial modeling by integrating fractional Brownian motion, long-memory effects, and regime-switching dynamics.

## Mathematical Framework

### Core Philosophy
- **Maximum Mathematical Sophistication**: Utilizes cutting-edge stochastic processes
- **Fractal Market Hypothesis**: Incorporates fractal geometry and long-memory effects
- **Behavioral Finance Integration**: Models market psychology and sentiment
- **Regime-Aware Dynamics**: Adapts to different market conditions

### Key Features
1. **Fractional Brownian Motion** with configurable Hurst exponent
2. **Long-memory effects** through historical correlation
3. **Mean reversion** with adaptive strength
4. **Regime switching** based on local volatility
5. **Multi-scale smoothing** with wavelet-inspired techniques
6. **Advanced probability mapping** with market psychology

## Engine Parameters

### Core Parameters
```python
sigma: float = 0.1                    # Base volatility (10% annualized)
scale: float = 10_000                 # Exposure sensitivity parameter
k: float = 0.45                       # Maximum probability deviation
T: float = 1 / (365 * 24)            # Time horizon (1 hour)
S_0: float = 100_000                  # Initial index price
dt: float = 1 / (86_400 * 365)       # Time step (1 second)
```

### Advanced Parameters
```python
mean_reversion_strength: float = 0.05  # Mean reversion intensity
fractal_dimension: float = 1.5         # Fractal roughness (1.5 = standard Brownian)
memory_length: int = 20                # Historical memory length
regime_threshold: float = 0.3          # Regime switching threshold
smoothness_factor: float = 2.0         # Probability mapping smoothness
```

### Derived Parameters
```python
hurst = 2 - fractal_dimension          # Hurst exponent (0.5 = standard Brownian)
```

## Mathematical Components

### 1. Exposure-to-Probability Mapping

The creative version uses a sophisticated multi-component approach:

```python
def exposure_to_probability(self, exposure: float) -> float:
    # Normalize exposure
    normalized_exposure = exposure / self.scale
    
    # Sigmoid component
    base_sigmoid = 1 / (1 + np.exp(-self.smoothness_factor * normalized_exposure))
    
    # Arctan component for smooth tails
    arctan_component = np.arctan(normalized_exposure) / np.pi + 0.5
    
    # Gaussian blending weight
    blend_weight = np.exp(-0.5 * normalized_exposure**2)
    blended_probability = (blend_weight * base_sigmoid + 
                          (1 - blend_weight) * arctan_component)
    
    # Market psychology factors
    if exposure >= 0:
        psychology_factor = np.tanh(normalized_exposure / 2)      # Gradual confidence
    else:
        psychology_factor = -np.tanh(-normalized_exposure / 1.5)  # Faster fear
    
    # Final adjustment
    adjusted_probability = blended_probability + 0.1 * psychology_factor
    scaled_probability = 0.5 + (adjusted_probability - 0.5) * (2 * self.k)
    
    return np.clip(scaled_probability, 0.5 - self.k, 0.5 + self.k)
```

**Mathematical Properties:**
- **Multi-component blend**: Sigmoid + Arctan + Psychology
- **Asymmetric response**: Different behavior for positive/negative exposure
- **Smooth transitions**: Continuous derivatives throughout
- **Bounded output**: Constrained to [0.5-k, 0.5+k] range

### 2. Probability-to-Drift Conversion

Advanced approach using adaptive distribution modeling:

```python
def compute_mu_from_probability(self, P: float) -> float:
    prob_distance = P - 0.5
    abs_prob_distance = abs(prob_distance)
    extremity_factor = np.tanh(4 * abs_prob_distance)
    
    # Adaptive distribution selection
    if extremity_factor < 0.5:
        d2 = norm.ppf(P)  # Normal distribution for moderate probabilities
    else:
        df = 5 - 3 * extremity_factor  # Adaptive degrees of freedom
        d2 = t.ppf(P, df)              # t-distribution for extreme probabilities
        t_adjustment = 1.0 + 0.2 * (1.0 - df / 5) * d2**2 / df
        d2 = d2 * t_adjustment
    
    # Base drift calculation
    base_mu = (d2 * self.sigma * np.sqrt(self.T) - np.log(S / K)) / self.T + 0.5 * self.sigma**2
    
    # Psychology-based scaling
    if prob_distance > 0:  # Bullish
        psychology_factor = 1.0 + np.sin(np.pi / 2 * prob_distance) * 0.5
    else:  # Bearish
        psychology_factor = 1.0 + np.sin(np.pi / 2 * abs_prob_distance) * 0.8
    
    scaled_mu = base_mu * psychology_factor
    non_linear_component = np.sign(scaled_mu) * (abs_prob_distance**1.5) * 10
    
    return scaled_mu + non_linear_component * 0.2
```

**Key Features:**
- **Adaptive distributions**: Normal for moderate, t-distribution for extreme probabilities
- **Fat tails**: Captures extreme market movements
- **Asymmetric psychology**: Different scaling for bullish vs bearish sentiment
- **Non-linear components**: Subtle variations for realism

### 3. Fractional Brownian Motion Price Generation

The most sophisticated component using fractional stochastic processes:

```python
def generate_price_path(self, mu: float, duration_in_seconds: int) -> np.ndarray:
    num_steps = int(duration_in_seconds / (self.dt * 86_400 * 365))
    price_path = np.zeros(num_steps + 1)
    price_path[0] = self.S_0
    
    # Initialize fractional process arrays
    increments = np.zeros(num_steps)
    regime_state = np.zeros(num_steps)
    
    # Generate correlated increments for fractional Brownian motion
    for i in range(num_steps):
        z = np.random.normal(0, 1)
        
        if i < self.memory_length:
            increments[i] = z  # Standard Brownian for initial steps
        else:
            # Apply fractional Brownian motion with memory
            memory_weights = np.power(
                np.arange(1, self.memory_length + 1), self.hurst - 1.5
            )
            memory_weights = memory_weights / np.sum(memory_weights)
            memory_effect = np.sum(
                memory_weights * increments[i - self.memory_length : i]
            )
            increments[i] = 0.7 * z + 0.3 * memory_effect
    
    # Generate path with regime awareness and mean reversion
    for i in range(num_steps):
        current_price = price_path[i]
        
        # Update regime state
        if i > 10:
            local_returns = np.diff(np.log(price_path[max(0, i - 10) : i + 1]))
            local_vol = np.std(local_returns) / np.sqrt(self.dt)
            target_regime = np.tanh((local_vol / self.sigma - 1) * 2)
            regime_state[i] = 0.9 * regime_state[i - 1] + 0.1 * target_regime
        
        # Adaptive volatility
        effective_sigma = self.sigma * (1 + 0.5 * regime_state[i])
        
        # Mean reversion with adaptive strength
        log_price_ratio = np.log(current_price / self.S_0)
        base_reversion = -self.mean_reversion_strength * log_price_ratio
        adaptive_factor = 1.0 + 0.5 * np.tanh(abs(log_price_ratio) - 0.1)
        mean_reversion = base_reversion * adaptive_factor
        
        # Combined drift
        effective_drift = mu + mean_reversion
        
        # Price update with fractional characteristics
        drift_term = (effective_drift - 0.5 * effective_sigma**2) * self.dt
        diffusion_term = effective_sigma * np.sqrt(self.dt) * increments[i]
        
        price_path[i + 1] = current_price * np.exp(drift_term + diffusion_term)
    
    return price_path
```

**Advanced Features:**
- **Memory effects**: Historical correlation through weighted past increments
- **Regime switching**: Dynamic volatility based on local market conditions
- **Adaptive mean reversion**: Stronger reversion when far from equilibrium
- **Fractal characteristics**: Controlled by Hurst exponent and fractal dimension

### 4. Multi-Scale Smoothing

Wavelet-inspired smoothing for dynamic exposure paths:

```python
def generate_dynamic_exposure_path(self, exposure_series: pd.Series) -> Tuple:
    # Multi-scale smoothing approach
    for i in range(n_steps):
        # Short-term window (recent changes)
        short_window = max(3, ma_window // 4)
        short_values = drift_path[max(0, i - short_window + 1) : i + 1]
        
        # Medium-term window (main smoothing)
        med_values = drift_path[max(0, i - ma_window + 1) : i + 1]
        
        # Long-term window (trend detection)
        long_window = min(ma_window * 2, n_steps)
        long_values = drift_path[max(0, i - long_window + 1) : i + 1]
        
        # Exponential weighting for each scale
        alpha_short, alpha_med, alpha_long = 0.3, 0.1, 0.05
        
        # Calculate weighted averages
        short_avg = exponential_weighted_average(short_values, alpha_short)
        med_avg = exponential_weighted_average(med_values, alpha_med)
        long_avg = exponential_weighted_average(long_values, alpha_long)
        
        # Adaptive blending based on volatility
        if i > 10:
            recent_drifts = drift_path[max(0, i - 10) : i + 1]
            drift_volatility = np.std(recent_drifts)
            vol_factor = np.tanh(drift_volatility * 5)
            
            short_weight = 0.5 + 0.3 * vol_factor
            med_weight = 0.3 - 0.1 * vol_factor
            long_weight = 0.2 - 0.1 * vol_factor
        else:
            short_weight, med_weight, long_weight = 0.5, 0.3, 0.2
        
        # Final smoothed value
        smoothed_drift_path[i] = (short_weight * short_avg + 
                                 med_weight * med_avg + 
                                 long_weight * long_avg)
```

**Smoothing Features:**
- **Multi-scale approach**: Short, medium, and long-term components
- **Adaptive weighting**: Volatility-based weight adjustment
- **Exponential decay**: Different decay rates for different scales
- **Volatility awareness**: More responsive during high volatility periods

## Validation and Metrics

### Advanced Validation Statistics

The creative version includes sophisticated validation metrics:

```python
def validate_statistics(self, price_path: np.ndarray, expected_mu: float) -> Dict:
    # Standard statistics
    log_returns = np.diff(np.log(price_path))
    realized_sigma = log_returns.std(ddof=1) / np.sqrt(self.dt)
    realized_mu = log_returns.mean() / self.dt + 0.5 * realized_sigma**2
    
    # Mean reversion measurement
    price_levels = price_path[:-1]
    price_level_norm = (price_levels - self.S_0) / self.S_0
    mean_reversion_corr = np.corrcoef(price_level_norm, log_returns)[0, 1]
    
    # Fractal dimension estimation using variance method
    scales = [1, 2, 4, 8, 16]
    variances = []
    
    for scale in scales:
        if scale < len(log_returns):
            agg_returns = np.array([
                np.sum(log_returns[i : i + scale])
                for i in range(0, len(log_returns) - scale + 1, scale)
            ])
            variances.append(np.var(agg_returns))
    
    # Estimate Hurst exponent
    if len(variances) > 1:
        log_scales = np.log(scales[:len(variances)])
        log_variances = np.log(variances)
        slope, _ = np.polyfit(log_scales, log_variances, 1)
        estimated_hurst = slope / 2
        estimated_fractal_dim = 2 - estimated_hurst
    else:
        estimated_fractal_dim = 1.5
    
    return {
        "realized_sigma": realized_sigma,
        "realized_mu": realized_mu,
        "expected_mu": expected_mu,
        "mu_valid": validate_direction_and_magnitude(price_path, expected_mu),
        "mean_reversion_corr": mean_reversion_corr,
        "fractal_dimension": estimated_fractal_dim,
        "total_return": (price_path[-1] / price_path[0]) - 1,
        # ... additional metrics
    }
```

### Fractal Analysis Metrics

- **Fractal Dimension**: Measures path roughness (1.0 = smooth, 2.0 = very rough)
- **Hurst Exponent**: Indicates long-term memory (0.5 = no memory, >0.5 = persistent)
- **Mean Reversion Strength**: Correlation between price level and subsequent returns
- **Regime Detection**: Identification of different volatility regimes

## Computational Characteristics

### Performance Profile
- **Computational Complexity**: Very High (O(n × m × s × r))
- **Memory Usage**: High (fractal arrays, regime states, memory buffers)
- **Processing Time**: Slow (complex calculations at each step)
- **Numerical Stability**: Good (but requires careful parameter tuning)

### Resource Requirements
- **CPU**: High utilization due to complex mathematical operations
- **Memory**: Significant for storing historical data and intermediate calculations
- **Storage**: Moderate for results and visualization data

## Use Cases and Applications

### Ideal Applications
1. **Academic Research**: Exploring fractal market hypothesis
2. **Advanced Quantitative Analysis**: Deep market microstructure studies
3. **Behavioral Finance Research**: Market psychology modeling
4. **High-Frequency Analysis**: Capturing complex temporal dynamics
5. **Risk Management Research**: Understanding extreme market behaviors

### Not Recommended For
1. **Production Trading Systems**: Too computationally intensive
2. **Real-time Applications**: Processing delays unacceptable
3. **Regulatory Reporting**: Too complex for compliance requirements
4. **Educational Purposes**: Overwhelming complexity for learning

## Advantages and Limitations

### Advantages
1. **Maximum Realism**: Most sophisticated market behavior modeling
2. **Fractal Characteristics**: Captures self-similar market patterns
3. **Long-Memory Effects**: Models persistent market trends
4. **Regime Awareness**: Adapts to changing market conditions
5. **Behavioral Integration**: Incorporates market psychology
6. **Research Value**: Cutting-edge financial modeling techniques

### Limitations
1. **Computational Intensity**: Very high processing requirements
2. **Parameter Complexity**: Many parameters requiring careful calibration
3. **Overfitting Risk**: Complex model may overfit to specific datasets
4. **Implementation Difficulty**: Requires deep mathematical understanding
5. **Validation Challenges**: Complex behavior difficult to validate
6. **Exploitation Vulnerability**: Predictable patterns may be exploitable

## Configuration Examples

### Research Configuration (Maximum Sophistication)
```python
engine = SupplyDemandIndexEngine(
    sigma=0.1,                        # Low base volatility
    scale=10_000,                     # High sensitivity
    k=0.45,                          # Wide probability range
    mean_reversion_strength=0.05,     # Moderate mean reversion
    fractal_dimension=1.6,           # Rougher than standard Brownian
    memory_length=20,                # Long memory
    regime_threshold=0.3,            # Sensitive regime switching
    smoothness_factor=2.0            # Smooth transitions
)
```

### High-Frequency Configuration
```python
engine = SupplyDemandIndexEngine(
    sigma=0.15,                      # Higher volatility for HF
    scale=5_000,                     # Very high sensitivity
    k=0.4,                          # Moderate probability range
    mean_reversion_strength=0.08,    # Stronger mean reversion
    fractal_dimension=1.7,          # More roughness
    memory_length=10,               # Shorter memory for responsiveness
    regime_threshold=0.2,           # More sensitive regime detection
    smoothness_factor=1.5           # Less smoothing for responsiveness
)
```

## Conclusion

The Creative Supply-Demand Index Engine represents the pinnacle of mathematical sophistication in supply-demand modeling. It incorporates cutting-edge stochastic processes, fractal geometry, and behavioral finance principles to create the most realistic market behavior simulation possible.

While computationally intensive and complex to implement, this engine provides unparalleled insights into market dynamics and serves as an excellent platform for advanced financial research and quantitative analysis.

**Key Takeaway**: Use this engine when maximum mathematical sophistication and market realism are priorities, and computational resources are abundant. It's the "research prototype" that pushes the boundaries of what's possible in financial modeling.
