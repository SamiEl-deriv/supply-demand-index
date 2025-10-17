# Adjusted Supply-Demand Index Engine Documentation

## Overview

The Adjusted Supply-Demand Index Engine implements an advanced mathematical approach to supply-demand index generation with sophisticated stochastic processes, smooth non-linear mappings, and market psychology integration.

## Mathematical Framework

### Core Philosophy
- **Smooth Transitions**: Uses advanced mathematical functions for continuous, differentiable mappings
- **Market Psychology**: Incorporates behavioral finance concepts into probability and drift calculations
- **Adaptive Scaling**: Dynamic sensitivity based on exposure magnitude
- **No Predictable Patterns**: Deliberately avoids features that could be exploited

## Class: SupplyDemandIndexEngine

### Initialization Parameters

```python
def __init__(
    sigma: float = 0.1,           # Base volatility (annualized)
    scale: float = 10_000,        # Exposure sensitivity parameter
    k: float = 0.45,              # Maximum deviation from 0.5 in probability
    T: float = 1 / (365 * 24),    # Time horizon (1 hour in years)
    S_0: float = 100_000,         # Initial index price
    dt: float = 1 / (86_400 * 365), # Time step (1 second in years)
    smoothness_factor: float = 2.0  # Controls transition smoothness
)
```

**Parameter Summary:**
- `sigma`: Controls the base volatility level for all simulations
- `scale`: Determines sensitivity of probability mapping to exposure changes
- `k`: Constrains probability range to [0.5-k, 0.5+k]
- `T`: Time horizon for probability calculations
- `S_0`: Starting price level for all simulations
- `dt`: Granularity of time steps in the simulation
- `smoothness_factor`: Controls the steepness of probability transitions

## Core Mathematical Functions

### 1. exposure_to_probability(exposure: float) → float

**Mathematical Approach:**
This function implements a sophisticated multi-component mapping that combines several mathematical functions for natural market behavior.

**Components:**
1. **Adaptive Scaling**: `normalized_exposure = exposure / scale`
2. **Sigmoid Transformation**: `base_sigmoid = 1 / (1 + exp(-smoothness_factor * normalized_exposure))`
3. **Arctan Component**: `arctan_component = arctan(normalized_exposure) / π + 0.5`
4. **Gaussian Blending**: `blend_weight = exp(-0.5 * normalized_exposure²)`
5. **Market Psychology Factor**:
   - Positive exposure: `psychology_factor = tanh(normalized_exposure / 2)`
   - Negative exposure: `psychology_factor = -tanh(-normalized_exposure / 1.5)`

**Final Formula:**
```
blended_probability = blend_weight * base_sigmoid + (1 - blend_weight) * arctan_component
adjusted_probability = blended_probability + 0.1 * psychology_factor
scaled_probability = 0.5 + (adjusted_probability - 0.5) * (2 * k)
```

**Goal:** Map net exposure to probability with smooth transitions and market psychology effects.

**Parameters:**
- `exposure`: Net exposure value (positive = bullish, negative = bearish)

**Returns:** Probability ∈ [0.5-k, 0.5+k]

### 2. compute_mu_from_probability(P: float, S: float = None, K: float = None) → float

**Mathematical Approach:**
Combines multiple financial theories with adaptive distribution modeling.

**Components:**
1. **Probability Distance**: `prob_distance = P - 0.5`
2. **Extremity Factor**: `extremity_factor = tanh(4 * |prob_distance|)`
3. **Distribution Blending**:
   - Moderate probabilities: Normal distribution quantile
   - Extreme probabilities: t-distribution with adaptive degrees of freedom
4. **Base Drift Calculation** (Black-Scholes derived):
   ```
   base_mu = (d2 * σ * √T - ln(S/K)) / T + 0.5 * σ²
   ```
5. **Psychology Scaling**:
   - Bullish: `psychology_factor = 1.0 + sin(π/2 * prob_distance) * 0.5`
   - Bearish: `psychology_factor = 1.0 + sin(π/2 * |prob_distance|) * 0.8`

**Goal:** Convert probability to drift parameter with market psychology and fat-tail behavior.

**Parameters:**
- `P`: Desired probability of ending above strike
- `S`: Current spot price (optional, defaults to S_0)
- `K`: Strike price (optional, defaults to S for "above current" probability)

**Returns:** Drift parameter μ with psychological scaling

### 3. generate_price_path(mu: float, duration_in_seconds: int, random_seed: Optional[int] = None) → np.ndarray

**Mathematical Approach:**
Standard Geometric Brownian Motion (GBM) implementation.

**Formula:**
```
dS = S * (μ * dt + σ * √dt * dW)
```
Where dW is a Wiener process (random normal).

**Implementation:**
```
log_return = (μ - σ²/2) * dt + σ * √dt * N(0,1)
S[t+1] = S[t] * exp(log_return)
```

**Goal:** Generate realistic price paths following GBM dynamics.

**Parameters:**
- `mu`: Drift parameter (annualized)
- `duration_in_seconds`: Simulation length
- `random_seed`: For reproducibility

**Returns:** Array of price levels

### 4. generate_dynamic_exposure_path(exposure_series: pd.Series, ...) → Tuple

**Mathematical Approach:**
Three-pass algorithm for dynamic price path generation.

**Pass 1 - Raw Drift Calculation:**
For each exposure value:
1. Map exposure → probability using `exposure_to_probability()`
2. Map probability → drift using `compute_mu_from_probability()`

**Pass 2 - Weighted Moving Average Smoothing:**
```python
alpha = 0.1  # Smoothing factor
weights = [(1 - alpha)^j for j in range(window_size-1, -1, -1)]
weights = weights / sum(weights)  # Normalize to sum to 1
smoothed_drift_path[i] = sum(window_values * weights)
```

**Pass 3 - Price Path Generation:**
Standard GBM with time-varying drift:
```
log_return = (μ[t] - σ²/2) * dt + σ * √dt * N(0,1)
```

**Goal:** Create price paths that respond immediately to exposure changes.

**Parameters:**
- `exposure_series`: Time series of exposure values
- `random_seed`: For reproducibility
- `ma_window`: Not used in adjusted version (no smoothing)

**Returns:** (price_path, drift_path, smoothed_drift_path, probability_path)

### 5. validate_statistics(price_path: np.ndarray, expected_mu: float) → Dict

**Mathematical Approach:**
Enhanced validation with direction and magnitude checks.

**Metrics Calculated:**
1. **Realized Volatility**: `σ_realized = std(log_returns) / √dt`
2. **Realized Drift**: `μ_realized = mean(log_returns) / dt + 0.5 * σ_realized²`
3. **Total Return**: `(S_final / S_initial) - 1`
4. **Direction Validation**:
   - μ > 0: Check if S_final > S_initial
   - μ < 0: Check if S_final < S_initial
   - μ ≈ 0: Check if |total_return| < 1%
5. **Magnitude Validation**: Check if actual return is within 50% of expected

**Goal:** Validate that simulated paths match expected statistical properties.

**Parameters:**
- `price_path`: Simulated price array
- `expected_mu`: Expected drift parameter

**Returns:** Dictionary with validation results and statistics

## Key Mathematical Innovations

### 1. Multi-Component Probability Mapping
- **Sigmoid Component**: Provides smooth S-curve behavior
- **Arctan Component**: Better tail behavior than pure sigmoid
- **Gaussian Blending**: Smooth transition between components
- **Psychology Factors**: Asymmetric response to positive/negative exposure

### 2. Adaptive Distribution Modeling
- **Normal Distribution**: For moderate probabilities
- **t-Distribution**: For extreme probabilities (fat tails)
- **Adaptive Degrees of Freedom**: Based on probability extremity

### 3. Market Psychology Integration
- **Bullish Sentiment**: Gradual confidence building (sin function)
- **Bearish Sentiment**: Faster fear response (steeper scaling)
- **Asymmetric Response**: Different scaling for positive vs negative exposure

### 4. No Smoothing Philosophy
- **Direct Response**: Each drift value responds immediately to exposure
- **No Memory**: Eliminates predictable patterns
- **Market Efficiency**: Prevents exploitation of smoothing artifacts

## Parameter Sensitivity Analysis

### High Impact Parameters:
- **scale**: Directly affects exposure sensitivity
- **smoothness_factor**: Controls transition steepness
- **k**: Constrains probability range

### Medium Impact Parameters:
- **sigma**: Affects volatility but not drift calculation
- **T**: Influences drift magnitude through Black-Scholes formula

### Low Impact Parameters:
- **S_0**: Only affects initial price level
- **dt**: Affects simulation granularity but not core behavior

## Mathematical Advantages

1. **Smooth Differentiability**: All functions are continuously differentiable
2. **Bounded Outputs**: Probability always in valid range [0.5-k, 0.5+k]
3. **Market Realism**: Psychology factors create realistic asymmetric responses
4. **Theoretical Foundation**: Based on established financial mathematics
5. **Adaptive Behavior**: Responds differently to different market regimes

## Limitations and Considerations

1. **Computational Complexity**: More complex than basic approaches
2. **Parameter Sensitivity**: Requires careful calibration
3. **No Mean Reversion**: Standard GBM doesn't include mean reversion
4. **Psychology Assumptions**: Market psychology factors are empirically derived
