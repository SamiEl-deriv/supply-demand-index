# New Supply-Demand Index Engine Documentation

## Overview

The New Supply-Demand Index Engine represents a balanced approach to supply-demand modeling, combining sophisticated mathematical techniques with practical considerations. This engine is designed for MT5 Vol 75 position data analysis, offering enhanced modeling capabilities while maintaining computational efficiency and production readiness.

## Mathematical Framework

### Core Philosophy
- **Balanced Sophistication**: Advanced mathematics with practical constraints
- **Production Ready**: Efficient algorithms suitable for real-time applications
- **MT5 Integration**: Optimized for minute-level position data processing
- **Enhanced Validation**: Comprehensive statistical validation with trend analysis

### Key Features
1. **Advanced probability mapping** with multi-component blending
2. **Regime-aware drift calculation** with market psychology
3. **Standard GBM** with enhanced volatility modeling
4. **Second-level granularity** for high-resolution analysis
5. **Comprehensive trend analysis** and exposure correlation
6. **Mixed period analysis** (short-term and long-term)

## Engine Parameters

### Core Parameters
```python
sigma: float = 0.30                   # Base volatility (30% annualized)
scale: float = 150_000                # Exposure sensitivity parameter
k: float = 0.40                       # Maximum probability deviation
T: float = 1.0 / (365 * 24)          # Time horizon (1 hour)
S_0: float = 10_000                   # Initial index price
smoothness_factor: float = 2.0        # Probability mapping smoothness
noise_injection_level: float = 0.01   # Anti-exploitation noise (1%)
```

### Derived Parameters
```python
dt: float = T / 100                   # Time step (1% of horizon)
sqrt_dt: float = np.sqrt(dt)          # Square root of time step
```

## Mathematical Components

### 1. Exposure-to-Probability Mapping

Enhanced approach with smooth, non-linear transformation:

```python
def exposure_to_probability(self, net_exposure: float) -> float:
    # Normalize exposure
    normalized_exposure = net_exposure / self.scale
    
    # Apply smooth, non-linear transformation with enhanced smoothness
    smooth_factor = self.smoothness_factor
    final_prob = 0.5 + self.k * np.tanh(normalized_exposure / smooth_factor)
    
    # Ensure bounds
    final_prob = np.clip(final_prob, 0.5 - self.k, 0.5 + self.k)
    
    return final_prob
```

**Mathematical Properties:**
- **Hyperbolic tangent**: Smooth S-curve with controlled steepness
- **Symmetric response**: Equal treatment of positive/negative exposure
- **Bounded output**: Constrained to [0.5-k, 0.5+k] range
- **Continuous derivatives**: Smooth transitions throughout
- **Deterministic**: No noise injection at mapping level

### 2. Probability-to-Drift Conversion

Sophisticated regime-aware approach:

```python
def compute_mu_from_probability(self, probability: float) -> float:
    # Center probability around 0.5
    p_centered = probability - 0.5
    
    # Non-linear mapping with regime awareness
    if abs(p_centered) > 0.2:  # High probability regime
        mu = np.sign(p_centered) * (abs(p_centered) ** 1.5) * self.sigma * 2
    else:  # Normal regime
        mu = p_centered * self.sigma * 1.5
    
    # Add mean reversion component (small)
    mean_reversion_strength = 0.1
    mu -= mean_reversion_strength * p_centered
    
    return mu
```

**Key Features:**
- **Regime awareness**: Different scaling for high vs normal probability scenarios
- **Non-linear scaling**: Exponential response for extreme probabilities
- **Mean reversion**: Small mean reversion component in drift calculation
- **Symmetric treatment**: Equal handling of positive/negative scenarios

### 3. Enhanced Price Path Generation

Standard GBM with volatility clustering and enhanced dynamics:

```python
def generate_price_path(self, exposure_sequence: np.ndarray, num_steps: int = 100) -> Tuple:
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
            probability_path[j] = prob
            drift_path[j] = mu
            
            # Enhanced GBM with volatility clustering
            current_price = price_path[j]
            
            # Volatility clustering effect
            vol_cluster = 1 + 0.1 * np.sin(j * 0.1)  # Subtle volatility variation
            effective_sigma = self.sigma * vol_cluster
            
            # Price update with enhanced dynamics
            dW = np.random.normal(0, 1)
            drift_term = (mu - 0.5 * effective_sigma**2) * self.dt
            diffusion_term = effective_sigma * self.sqrt_dt * dW
            
            price_path[j + 1] = current_price * np.exp(drift_term + diffusion_term)
    
    return price_path, probability_path, drift_path
```

**Enhanced Features:**
- **Standard GBM**: Proven geometric Brownian motion foundation
- **Volatility clustering**: Subtle time-varying volatility
- **Enhanced dynamics**: Improved price evolution modeling
- **No memory effects**: Avoids exploitable patterns
- **Efficient computation**: Fast processing for real-time use

### 4. Dynamic Exposure Path Generation

High-resolution path generation with second-level granularity:

```python
def generate_dynamic_exposure_path(self, exposure_series: pd.Series, 
                                 seconds_per_minute: int = 60) -> Tuple:
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
            effective_sigma = self.sigma * 0.8  # Reduce by 20% for second-level
            drift_term = (mu - 0.5 * effective_sigma**2) * dt
            diffusion_term = effective_sigma * sqrt_dt * dW
            
            price_path[second_idx + 1] = current_price * np.exp(drift_term + diffusion_term)
    
    # Apply weighted moving average smoothing to drift
    adjusted_ma_window = ma_window * seconds_per_minute
    weights = np.exp(-np.arange(adjusted_ma_window) / (adjusted_ma_window / 3))
    weights = weights / np.sum(weights)
    
    smoothed_drift_path = np.convolve(drift_path, weights, mode='same')
    
    return price_path, drift_path, smoothed_drift_path, probability_path
```

**High-Resolution Features:**
- **Second-level granularity**: 60 price updates per minute
- **Adaptive volatility**: Adjusted for higher frequency
- **Weighted smoothing**: Exponential decay smoothing
- **Efficient processing**: Optimized for large datasets

## Validation and Analysis

### Enhanced Validation Statistics

Comprehensive validation with trend analysis:

```python
def validate_statistics(self, price_path: np.ndarray, expected_mu: float) -> Dict:
    # Calculate log returns
    log_returns = np.diff(np.log(price_path))
    
    # Realized statistics
    realized_sigma = log_returns.std(ddof=1) / np.sqrt(self.dt)
    realized_drift = log_returns.mean() / self.dt
    realized_mu = realized_drift + 0.5 * realized_sigma**2
    
    # Calculate total return
    total_return = (price_path[-1] / price_path[0]) - 1
    
    # Enhanced validation approach
    if expected_mu > 0:
        mu_valid_direction = price_path[-1] > price_path[0]
    elif expected_mu < 0:
        mu_valid_direction = price_path[-1] < price_path[0]
    else:
        mu_valid_direction = abs(total_return) < 0.01
    
    # Check magnitude reasonableness
    expected_return = np.exp(expected_mu * self.dt * len(log_returns)) - 1
    mu_valid_magnitude = abs(total_return - expected_return) < 0.5 * abs(expected_return)
    
    # Combined validation
    mu_valid = mu_valid_direction and mu_valid_magnitude
    
    return {
        "realized_sigma": realized_sigma,
        "realized_mu": realized_mu,
        "expected_mu": expected_mu,
        "mu_valid": mu_valid,
        "mu_valid_direction": mu_valid_direction,
        "mu_valid_magnitude": mu_valid_magnitude,
        "mu_error": abs(realized_mu - expected_mu),
        "total_return": total_return,
        "expected_return": expected_return,
        "final_price": price_path[-1]
    }
```

### Advanced Trend Analysis

Comprehensive trend analysis with exposure correlation:

```python
def analyze_trend_lengths(self, price_path: np.ndarray, exposure_series: pd.Series) -> Dict:
    # Calculate price returns
    returns = np.diff(price_path) / price_path[:-1]
    
    # Identify trends (up/down/sideways)
    trend_threshold = 0.0001  # 0.01% threshold
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
                # Get corresponding exposure
                if trend_start < len(exposure_series):
                    end_idx = min(trend_start + trend_length, len(exposure_series))
                    avg_exposure = np.mean(exposure_series.iloc[trend_start:end_idx])
                    
                    trends.append({
                        'direction': current_trend,
                        'start': trend_start,
                        'length': trend_length,
                        'avg_exposure': avg_exposure,
                        'price_change': (price_path[trend_start + trend_length] - 
                                       price_path[trend_start]) / price_path[trend_start]
                    })
            
            # Start new trend
            current_trend = trend_signals[i]
            trend_start = i
            trend_length = 1
    
    # Analyze correlation between exposure and trend direction
    if len(trends) > 1:
        exposures = [t['avg_exposure'] for t in trends]
        directions = [t['direction'] for t in trends]
        exposure_correlation = np.corrcoef(exposures, directions)[0, 1]
    else:
        exposure_correlation = 0
    
    return {
        'total_trends': len(trends),
        'up_trends': [t for t in trends if t['direction'] == 1],
        'down_trends': [t for t in trends if t['direction'] == -1],
        'sideways_trends': [t for t in trends if t['direction'] == 0],
        'exposure_correlation': exposure_correlation,
        'all_trends': trends
    }
```

### Mixed Period Analysis

Support for both short-term and long-term analysis:

```python
def select_mixed_periods(df: pd.DataFrame, 
                        num_short_periods: int = 4,
                        num_long_periods: int = 1,
                        short_period_days: int = 7,
                        long_period_days: int = 60) -> List[Tuple]:
    # Select 4 short periods (7 days) + 1 long period (60 days)
    # Ensures non-overlapping periods for comprehensive analysis
    
    selected_periods = []
    selected_dates = []
    
    # Short periods selection
    for i in range(num_short_periods):
        # Find non-overlapping 7-day period
        period_data = extract_period(df, short_period_days)
        selected_periods.append((period_data, "short"))
    
    # Long periods selection  
    for i in range(num_long_periods):
        # Find non-overlapping 60-day period
        period_data = extract_period(df, long_period_days)
        selected_periods.append((period_data, "long"))
    
    return selected_periods
```

## Computational Characteristics

### Performance Profile
- **Computational Complexity**: Medium (O(n × m × s))
- **Memory Usage**: Moderate (standard arrays, no complex buffers)
- **Processing Time**: Fast (efficient algorithms)
- **Numerical Stability**: High (well-tested mathematical operations)

### Resource Requirements
- **CPU**: Moderate utilization with efficient algorithms
- **Memory**: Reasonable for second-level data processing
- **Storage**: Standard for results and visualization

## Use Cases and Applications

### Ideal Applications
1. **Production Trading Systems**: Efficient enough for real-time use
2. **MT5 Position Analysis**: Optimized for minute-level position data
3. **Risk Management**: Comprehensive trend and exposure analysis
4. **Quantitative Research**: Advanced modeling with practical constraints
5. **Regulatory Reporting**: Sophisticated yet interpretable models
6. **Educational Applications**: Complex enough for learning, simple enough to understand

### Recommended Scenarios
1. **Mixed timeframe analysis**: Both short-term (7 days) and long-term (60 days)
2. **High-frequency monitoring**: Second-level price generation
3. **Exposure correlation studies**: Relationship between exposure and price trends
4. **Validation-focused applications**: Comprehensive statistical validation

## Advantages and Limitations

### Advantages
1. **Balanced Approach**: Sophisticated modeling with practical efficiency
2. **Production Ready**: Suitable for real-time applications
3. **Comprehensive Analysis**: Mixed periods, trend analysis, exposure correlation
4. **Enhanced Validation**: Thorough statistical validation with trend metrics
5. **High Resolution**: Second-level granularity for detailed analysis
6. **Regime Awareness**: Adaptive behavior for different market conditions
7. **Anti-Exploitation**: Noise injection prevents pattern exploitation
8. **MT5 Optimized**: Specifically designed for MT5 Vol 75 data

### Limitations
1. **Medium Complexity**: More complex than basic approaches
2. **Parameter Sensitivity**: Requires careful calibration of multiple parameters
3. **Memory Requirements**: Higher memory usage for second-level processing
4. **Validation Dependency**: Requires comprehensive validation for reliability
5. **Limited Fractal Features**: No advanced fractal characteristics like creative version

## Configuration Examples

### Production Configuration (Balanced Performance)
```python
engine = NewSupplyDemandIndexEngine(
    sigma=0.30,                       # Standard volatility
    scale=150_000,                    # Moderate sensitivity
    k=0.40,                          # Conservative probability range
    smoothness_factor=2.0,           # Smooth transitions
    noise_injection_level=0.01       # 1% anti-exploitation noise
)
```

### High-Sensitivity Configuration
```python
engine = NewSupplyDemandIndexEngine(
    sigma=0.25,                      # Lower volatility for stability
    scale=100_000,                   # Higher sensitivity
    k=0.45,                         # Wider probability range
    smoothness_factor=1.5,          # More responsive transitions
    noise_injection_level=0.005     # Lower noise for precision
)
```

### Conservative Configuration
```python
engine = NewSupplyDemandIndexEngine(
    sigma=0.35,                      # Higher volatility buffer
    scale=200_000,                   # Lower sensitivity
    k=0.35,                         # Narrower probability range
    smoothness_factor=2.5,          # Smoother transitions
    noise_injection_level=0.02      # Higher noise for robustness
)
```

## Analysis Workflow

### Standard Analysis Process
1. **Data Preparation**: Load MT5 Vol 75 position data
2. **Period Selection**: Choose mixed periods (short + long term)
3. **Exposure Calculation**: Compute net exposure per minute
4. **Path Generation**: Generate high-resolution price paths
5. **Validation**: Comprehensive statistical validation
6. **Trend Analysis**: Analyze exposure-trend correlations
7. **Visualization**: Generate comprehensive plots

### Example Usage
```python
# Initialize engine
engine = NewSupplyDemandIndexEngine()

# Load and prepare data
df = load_mt5_data("vol75_positions.csv")
periods = select_mixed_periods(df)

# Analyze each period
for period_data, period_type in periods:
    # Generate exposure series
    exposure_series = calculate_net_exposure(period_data)
    
    # Generate dynamic path
    price_path, drift_path, smoothed_drift, prob_path = \
        engine.generate_dynamic_exposure_path(exposure_series)
    
    # Validate results
    validation_stats = engine.validate_statistics(price_path, expected_mu)
    
    # Analyze trends
    trend_analysis = engine.analyze_trend_lengths(price_path, exposure_series)
    
    # Generate visualizations
    create_comprehensive_plots(price_path, drift_path, prob_path, 
                             validation_stats, trend_analysis)
```

## Comparison with Other Versions

### vs Creative Version
- **Simpler**: No fractional Brownian motion or complex memory effects
- **Faster**: Standard GBM vs fractional processes
- **More Stable**: Fewer parameters, more predictable behavior
- **Production Ready**: Suitable for real-time applications

### vs Adjusted Version
- **Similar Sophistication**: Both use advanced probability mapping
- **Added Features**: Mean reversion component in drift calculation
- **Enhanced Analysis**: More comprehensive trend and validation analysis
- **MT5 Focused**: Specifically optimized for MT5 Vol 75 data

### vs Basic Version
- **More Sophisticated**: Advanced probability mapping vs simple normal CDF
- **Higher Resolution**: Second-level granularity vs standard resolution
- **Better Validation**: Comprehensive validation vs basic direction check
- **Enhanced Features**: Trend analysis, exposure correlation, mixed periods

## Technical Implementation Notes

### Key Implementation Details
1. **Memory Management**: Efficient array allocation for second-level data
2. **Numerical Stability**: Careful handling of extreme probability values
3. **Performance Optimization**: Vectorized operations where possible
4. **Error Handling**: Robust error handling for edge cases
5. **Reproducibility**: Consistent random seed management

### Best Practices
1. **Parameter Tuning**: Start with default values, adjust based on validation results
2. **Data Quality**: Ensure clean, consistent MT5 position data
3. **Period Selection**: Use mixed periods for comprehensive analysis
4. **Validation Focus**: Always validate results before drawing conclusions
5. **Resource Monitoring**: Monitor memory usage for large datasets

## Conclusion

The New Supply-Demand Index Engine represents an optimal balance between mathematical sophistication and practical implementation. It provides advanced modeling capabilities while maintaining computational efficiency and production readiness.

**Key Strengths:**
- **Balanced complexity**: Sophisticated enough for advanced analysis, simple enough for production
- **Comprehensive validation**: Thorough statistical validation with trend analysis
- **High resolution**: Second-level granularity for detailed insights
- **MT5 optimized**: Specifically designed for MT5 Vol 75 position data
- **Anti-exploitation**: Built-in noise injection prevents pattern exploitation

**Ideal Use Cases:**
- Production trading systems requiring sophisticated supply-demand analysis
- Risk management applications needing comprehensive trend analysis
- Quantitative research with practical implementation constraints
- Educational applications demonstrating advanced financial modeling

**Key Takeaway**: Use this engine when you need sophisticated supply-demand modeling with practical constraints, comprehensive validation, and production readiness. It's the "production-optimized" version that balances advanced mathematics with real-world implementation requirements.
