# Basic vs Adjusted Supply-Demand Index: Mathematical Comparison

## Executive Summary

This document provides a comprehensive mathematical comparison between the Basic and Adjusted Supply-Demand Index approaches, analyzing their theoretical foundations, parameter configurations, computational characteristics, and practical implications.

## Overview of Approaches

### Basic Approach Philosophy
- **Mathematical Rigor**: Established financial mathematics (Black-Scholes, normal distributions)
- **Theoretical Soundness**: Proven mathematical relationships
- **Computational Efficiency**: Standard algorithms with predictable behavior
- **Stability**: Consistent, smooth responses through traditional smoothing

### Adjusted Approach Philosophy
- **Advanced Modeling**: Sophisticated mathematical functions with market psychology
- **Behavioral Finance**: Integration of market sentiment and psychology factors
- **High Resolution Processing**: Per-second data processing with smoothing
- **Adaptive Behavior**: Dynamic sensitivity based on market conditions

## Mathematical Framework Comparison

### 1. Exposure-to-Probability Mapping

#### Basic Approach: Normal CDF cumulative distribution function 
```
normalized_exposure = exposure / scale
base_probability = Φ(normalized_exposure)  # Standard normal CDF
scaled_probability = 0.5 + (base_probability - 0.5) * (2 * k)
```

**Mathematical Properties:**
- **Function**: Standard normal cumulative distribution function
- **Symmetry**: Perfect symmetry around zero exposure
- **Smoothness**: Continuously differentiable
- **Theoretical Foundation**: Established probability theory

#### Adjusted Approach: Multi-Component Blend
```
# Sigmoid component
base_sigmoid = 1 / (1 + exp(-smoothness_factor * normalized_exposure))

# Arctan component  
arctan_component = arctan(normalized_exposure) / π + 0.5

# Gaussian blending
blend_weight = exp(-0.5 * normalized_exposure²)
blended_probability = blend_weight * base_sigmoid + (1 - blend_weight) * arctan_component

# Market psychology
psychology_factor = tanh(normalized_exposure / factor)  # Different factors for +/-
adjusted_probability = blended_probability + 0.1 * psychology_factor
```

**Mathematical Properties:**
- **Function**: Composite of sigmoid, arctan, and hyperbolic tangent
- **Asymmetry**: Different responses to positive/negative exposure
- **Smoothness**: Multiple smoothness controls
- **Behavioral Foundation**: Market psychology integration

**Comparison:**
| Aspect | Basic (Normal CDF) | Adjusted (Multi-Component) |
|--------|-------------------|---------------------------|
| **Complexity** | Simple, single function | Complex, multiple components |
| **Symmetry** | Perfectly symmetric | Asymmetric (psychology factors) |
| **Parameters** | 2 (scale, k) | 4+ (scale, k, smoothness_factor, psychology factors) |
| **Interpretability** | High (standard statistics) | Medium (composite function) |
| **Computational Cost** | Low | High |
| **Market Realism** | Mathematical idealization | Behavioral realism |

### 2. Probability-to-Drift Conversion

#### Basic Approach: Black-Scholes with Scaling
```
d₂ = Φ⁻¹(P)  # Inverse normal CDF
base_mu = [d₂ * σ * √T - ln(S/K)] / T + 0.5σ²

# Linear scaling
prob_distance = |P - 0.5| / 0.5
scaling_factor = 1.7
scaled_mu = base_mu * (1 + scaling_factor * prob_distance)
```

**Mathematical Properties:**
- **Foundation**: Direct Black-Scholes relationship
- **Scaling**: Linear scaling factor
- **Symmetry**: Equal treatment of positive/negative probabilities
- **Predictability**: Deterministic relationship

#### Adjusted Approach: Multi-Distribution with Psychology
```
# Adaptive distribution selection
extremity_factor = tanh(4 * |prob_distance|)

if extremity_factor < 0.5:
    d₂ = norm.ppf(P)  # Normal distribution
else:
    df = 5 - 3 * extremity_factor  # Adaptive degrees of freedom
    d₂ = t.ppf(P, df)  # t-distribution with fat tails

# Psychology scaling
if bullish_sentiment:
    psychology_factor = 1.0 + sin(π/2 * prob_distance) * 0.5
else:
    psychology_factor = 1.0 + sin(π/2 * |prob_distance|) * 0.8

scaled_mu = base_mu * psychology_factor
```

**Mathematical Properties:**
- **Foundation**: Adaptive distribution modeling (normal + t-distribution)
- **Scaling**: Non-linear psychology-based scaling
- **Asymmetry**: Different treatment of bullish/bearish sentiment
- **Complexity**: Multiple distribution models

**Comparison:**
| Aspect | Basic (Black-Scholes) | Adjusted (Multi-Distribution) |
|--------|----------------------|------------------------------|
| **Distribution** | Normal only | Normal + t-distribution |
| **Fat Tails** | No | Yes (for extreme probabilities) |
| **Psychology** | None | Bullish/bearish asymmetry |
| **Scaling** | Linear | Non-linear (trigonometric) |
| **Parameters** | 1 (scaling_factor) | 3+ (extremity, psychology factors) |
| **Predictability** | High | Medium (adaptive behavior) |

### 3. Smoothing and Temporal Processing

#### Basic Approach: Exponential Weighted Moving Average
```
alpha = 0.1  # Fixed smoothing factor
weights = [(1 - alpha)^j for j in range(window_size-1, -1, -1)]
weights = weights / sum(weights)
smoothed_drift[i] = sum(window_values * weights)
```

**Mathematical Properties:**
- **Method**: Traditional EWMA with exponential decay
- **Memory**: Incorporates all historical values with decreasing weights
- **Stability**: Reduces noise, provides smooth transitions
- **Lag**: Introduces delay in response to changes

#### Adjusted Approach: Exponential Weighted Moving Average (Same as Basic)
```
alpha = 0.1  # Smoothing factor
weights = [(1 - alpha)^j for j in range(window_size-1, -1, -1)]
weights = weights / sum(weights)  # Normalize to sum to 1
smoothed_drift[i] = sum(window_values * weights)
```

**Mathematical Properties:**
- **Method**: Traditional EWMA with exponential decay (same as basic)
- **Memory**: Incorporates all historical values with decreasing weights
- **Stability**: Reduces noise, provides smooth transitions
- **Lag**: Introduces delay in response to changes

**Comparison:**
| Aspect | Basic (EWMA) | Adjusted (EWMA) |
|--------|-------------|-----------------|
| **Smoothing** | Yes (exponential weights) | Yes (exponential weights) |
| **Lag** | Present (smoothing lag) | Present (smoothing lag) |
| **Stability** | High (noise reduction) | High (noise reduction) |
| **Memory** | Historical memory | Historical memory |
| **Responsiveness** | Delayed | Delayed |
| **Predictability** | High (smooth changes) | High (smooth changes) |

### 4. Resolution and Processing

#### Basic Approach: Standard Resolution
```
window_minutes = 5           # 5-minute exposure windows
seconds_per_point = 300      # 5-minute intervals
ma_window = 12              # 1-hour moving average
```

#### Adjusted Approach: Same Resolution as Basic
```
window_minutes = 5           # 5-minute exposure windows (same as basic)
seconds_per_point = 1        # Per-second interpolation
ma_window = 12              # 1-hour moving average (same as basic)
```

**Comparison:**
| Aspect | Basic | Adjusted |
|--------|-------|----------|
| **Time Resolution** | 5 minutes | 1 second |
| **Data Points** | 12 per hour | 3,600 per hour |
| **Computational Load** | Low | High |
| **Memory Usage** | Moderate | High |
| **Temporal Precision** | Standard | Maximum |

## Parameter Configuration Comparison

### Engine Parameters

| Parameter | Basic Value | Adjusted Value | Impact |
|-----------|-------------|----------------|---------|
| **sigma** | 0.75 (75%) | 0.3 (30%) | Volatility level |
| **scale** | 100,000 | 150,000 | Exposure sensitivity |
| **k** | 0.45 | 0.4 | Probability range |
| **T** | 1/(365×24) | 1/(365×24) | Time horizon (same) |
| **S_0** | 100,000 | 100,000 | Initial price (same) |
| **dt** | 1/(86400×365) | 1/(86400×365) | Time step (same) |
| **smoothness_factor** | N/A | 2.0 | Transition control |

### Analysis Parameters

| Parameter | Basic Value | Adjusted Value | Purpose |
|-----------|-------------|----------------|---------|
| **window_minutes** | 5 | 5 | Exposure aggregation |
| **sample_size** | 50 | 50 | Analysis points |
| **num_simulations** | 100 | 20 | Monte Carlo paths |
| **ma_window** | 12 | 12 | Moving average window |
| **random_seed** | 1505 | 2024 | Reproducibility |

## Complete Parameter Reference Table

| Parameter | Description | Basic Value | Adjusted Value | Impact/Purpose |
|-----------|-------------|-------------|----------------|----------------|
| **Engine Core Parameters** |
| `sigma` | Volatility parameter for stochastic process | 0.75 (75%) | 0.3 (30%) | Controls price movement volatility; lower = smoother paths |
| `scale` | Exposure normalization factor | 100,000 | 150,000 | Scales exposure values for probability mapping; higher = less sensitive |
| `k` | Maximum probability deviation from neutral | 0.45 | 0.4 | Limits probability range to [0.5-k, 0.5+k]; lower = more conservative |
| `T` | Time horizon for drift calculations | 1/(365×24) | 1/(365×24) | 1-hour time horizon for Black-Scholes calculations |
| `S_0` | Initial/reference price level | 100,000 | 100,000 | Starting price for Monte Carlo simulations |
| `dt` | Time step for stochastic process | 1/(86400×365) | 1/(86400×365) | 1-second time steps for path generation |
| `smoothness_factor` | Controls transition smoothness | N/A | 2.0 | Adjusted only: controls sigmoid steepness in probability mapping |
| **Data Processing Parameters** |
| `window_minutes` | Time window for exposure aggregation | 5 | 5 | Groups trades into 5-minute exposure buckets |
| `seconds_per_point` | Temporal resolution for interpolation | 300 (5min) | 1 | Basic: 5-min intervals; Adjusted: per-second interpolation |
| `ma_window` | Moving average window size | 12 | 12 | Number of historical points for EWMA smoothing (1 hour) |
| `alpha` | EWMA smoothing factor | 0.1 | 0.1 | Exponential decay rate; lower = more smoothing |
| **Analysis Parameters** |
| `sample_size` | Number of exposure points for analysis | 50 | 50 | Limits computational load while maintaining statistical validity |
| `num_simulations` | Monte Carlo simulation count | 100 | 20 | Trade-off between accuracy and computation time |
| `num_paths_per_exposure` | Paths generated per exposure scenario | 50 | 50 | Monte Carlo paths for each exposure level |
| `duration_in_seconds` | Simulation duration | 3600 | 3600 | 1-hour simulation length for each scenario |
| `random_seed` | Base seed for reproducibility | 1505 | 2024 | Ensures consistent results across runs |
| **Visualization Parameters** |
| `plots_dir` | Output directory for plots | "plots/basic" | "plots/adjusted" | Separate directories to avoid overwriting |
| `dpi` | Plot resolution | 300 | 300 | High-resolution plots for publication quality |
| `figsize` | Default figure size | (12,8) | (16,10) | Larger plots for adjusted approach complexity |
| **Advanced Parameters (Adjusted Only)** |
| `psychology_factors` | Market sentiment scaling | N/A | [0.3, 0.5] | Asymmetric scaling for bullish/bearish scenarios |
| `extremity_threshold` | Distribution switching threshold | N/A | 0.5 | When to switch from normal to t-distribution |
| `min_df` | Minimum degrees of freedom for t-dist | N/A | 2 | Prevents extreme fat-tail behavior |
| `max_df` | Maximum degrees of freedom for t-dist | N/A | 5 | Controls maximum tail thickness |
| `blend_weight_factor` | Gaussian blending control | N/A | 0.5 | Controls sigmoid vs arctan blending in probability mapping |

## Computational Complexity Analysis

### Time Complexity

#### Basic Approach
- **Exposure Mapping**: O(n) - single function evaluation
- **Drift Calculation**: O(n) - single inverse CDF
- **Smoothing**: O(n × w) - window-based EWMA
- **Path Generation**: O(n × m × s) - standard GBM
- **Overall**: O(n × m × s) where n=exposures, m=simulations, s=steps

#### Adjusted Approach
- **Exposure Mapping**: O(n) - multiple function evaluations
- **Drift Calculation**: O(n) - adaptive distribution selection
- **Smoothing**: O(n × w) - window-based EWMA (same as basic)
- **Path Generation**: O(n × m × s) - standard GBM
- **Interpolation**: O(n × r) - high-resolution interpolation
- **Overall**: O(n × m × s × r) where r=resolution multiplier

### Space Complexity

#### Basic Approach
- **Data Storage**: O(n) - standard resolution
- **Simulation Results**: O(n × m) - moderate memory
- **Visualization**: O(n) - standard arrays

#### Adjusted Approach
- **Data Storage**: O(n × r) - high-resolution data
- **Simulation Results**: O(n × m × r) - high memory
- **Visualization**: O(n × r) - large arrays

## Performance Characteristics

### Computational Performance

| Metric | Basic | Adjusted | Ratio |
|--------|-------|----------|-------|
| **Processing Time** | Fast | Slow | ~10-100x |
| **Memory Usage** | Low | High | ~50-300x |
| **CPU Utilization** | Low | High | ~10-50x |
| **I/O Requirements** | Minimal | Moderate | ~5-10x |

### Statistical Performance

| Metric | Basic | Adjusted | Trade-off |
|--------|-------|----------|-----------|
| **Stability** | High | High | Both use smoothing |
| **Responsiveness** | Medium | Medium | Both have smoothing lag |
| **Noise Level** | Low | Low | Both use filtering |
| **Predictability** | High | Medium | Consistency vs adaptability |

## Mathematical Advantages and Disadvantages

### Basic Approach

#### Advantages
1. **Theoretical Rigor**: Based on established financial mathematics
2. **Computational Efficiency**: Fast processing with standard algorithms
3. **Predictable Behavior**: Consistent, interpretable results
4. **Parameter Simplicity**: Fewer parameters to calibrate
5. **Stability**: Smooth, noise-reduced behavior
6. **Mathematical Transparency**: Clear relationships and interpretability

#### Disadvantages
1. **Limited Realism**: No market psychology or behavioral factors
2. **Symmetric Response**: Equal treatment of bullish/bearish scenarios
3. **Fixed Parameters**: No adaptive behavior
4. **Smoothing Lag**: Delayed response to rapid changes
5. **Standard Resolution**: May miss high-frequency dynamics
6. **Linear Relationships**: Fixed scaling throughout

### Adjusted Approach

#### Advantages
1. **Market Realism**: Incorporates behavioral finance and psychology
2. **High Responsiveness**: Immediate reaction to exposure changes
3. **Adaptive Behavior**: Different responses to different market conditions
4. **Asymmetric Response**: Realistic bullish/bearish differentiation
5. **High Resolution**: Captures fine-grained temporal dynamics
6. **Advanced Mathematics**: Sophisticated modeling techniques

#### Disadvantages
1. **Computational Complexity**: High processing and memory requirements
2. **Parameter Sensitivity**: Many parameters requiring careful calibration
3. **High Resolution Overhead**: Per-second processing creates computational burden
4. **Lower Predictability**: More complex, less interpretable behavior
5. **Implementation Complexity**: More difficult to understand and validate
6. **Overfitting Risk**: Complex models may overfit to specific data

## Use Case Recommendations

### Choose Basic Approach When:
1. **Computational Resources are Limited**: Low processing power or memory
2. **Stability is Priority**: Need consistent, predictable behavior
3. **Interpretability is Important**: Need clear, explainable results
4. **Real-time Processing**: Need fast response times
5. **Regulatory Requirements**: Need mathematically transparent models
6. **Educational/Research**: Learning or teaching fundamental concepts

### Choose Adjusted Approach When:
1. **Market Realism is Priority**: Need behavioral finance integration
2. **High Responsiveness Required**: Need immediate reaction to changes
3. **Computational Resources Available**: Have sufficient processing power
4. **Advanced Analytics**: Need sophisticated modeling capabilities
5. **Research and Development**: Exploring advanced techniques
6. **High-Frequency Applications**: Need fine-grained temporal resolution

## Validation and Testing Comparison

### Basic Approach Validation
- **Direction Validation**: Simple check if drift direction matches price movement
- **Statistical Consistency**: Realized statistics vs theoretical expectations
- **Parameter Sensitivity**: Predictable response to parameter changes
- **Convergence Testing**: Monte Carlo convergence with sufficient simulations

### Adjusted Approach Validation
- **Direction and Magnitude**: Both direction and magnitude validation
- **Psychology Validation**: Check if behavioral factors work as expected
- **Responsiveness Testing**: Validate immediate response to exposure changes
- **Complexity Validation**: Ensure complex model doesn't overfit

## Conclusion

The choice between Basic and Adjusted approaches represents a fundamental trade-off between **simplicity/efficiency** and **sophistication/realism**:

### Basic Approach: "Mathematical Rigor"
- Emphasizes established financial theory
- Prioritizes computational efficiency and stability
- Suitable for production systems requiring predictable behavior
- Ideal for educational and regulatory contexts

### Adjusted Approach: "Market Realism"
- Emphasizes behavioral finance and market psychology
- Prioritizes responsiveness and sophisticated modeling
- Suitable for research and advanced analytics
- Ideal for high-frequency, high-resource environments

Both approaches have their place in the financial modeling ecosystem, and the choice depends on specific requirements regarding computational resources, model interpretability, market realism, and application context.
