# Basic Supply-Demand Index Engine Documentation

## Overview

The Basic Supply-Demand Index Engine implements a mathematically rigorous approach to supply-demand index generation using established financial mathematics, including normal CDF-based exposure mapping, Black-Scholes framework for drift calculation, and standard GBM price path generation.

## Mathematical Framework

### Core Philosophy
- **Mathematical Rigor**: Uses well-established financial mathematics and statistical theory
- **Theoretical Soundness**: Based on proven mathematical relationships (Black-Scholes, normal distributions)
- **Consistent Scaling**: Fixed scaling factors throughout the system
- **Standard Smoothing**: Traditional exponential weighted moving averages

## Class: SupplyDemandIndexEngine

### Initialization Parameters

```python
def __init__(
    sigma: float = 0.1,           # Fixed volatility (annualized)
    scale: float = 10_000,        # Exposure sensitivity parameter
    k: float = 0.45,              # Maximum deviation from 0.5 in probability
    T: float = 1 / (365 * 24),    # Time horizon (1 hour in years)
    S_0: float = 100_000,         # Initial index price
    dt: float = 1 / (86_400 * 365) # Time step (1 second in years)
)
```

**Parameter Summary:**
- `sigma`: Fixed volatility level for all simulations (no adaptive volatility)
- `scale`: Controls sensitivity of exposure-to-probability mapping
- `k`: Constrains probability range to [0.5-k, 0.5+k]
- `T`: Time horizon for Black-Scholes probability calculations
- `S_0`: Starting price level for all simulations
- `dt`: Time step granularity in the simulation

## Core Mathematical Functions

### 1. exposure_to_probability(exposure: float) → float

**Mathematical Approach:**
Uses the cumulative distribution function (CDF) of a standard normal distribution for theoretically sound exposure-to-probability mapping.

**Formula:**
```
normalized_exposure = exposure / scale
base_probability = Φ(normalized_exposure)  # Standard normal CDF
scaled_probability = 0.5 + (base_probability - 0.5) * (2 * k)
```

Where Φ(x) is the standard normal CDF: `Φ(x) = ∫_{-∞}^x (1/√(2π)) * e^(-t²/2) dt`

**Mathematical Properties:**
1. **Smooth S-Curve**: The normal CDF provides a smooth, continuous mapping
2. **Symmetric**: Equal response to positive and negative exposures
3. **Bounded**: Output always in range [0, 1] before scaling
4. **Well-Defined**: Based on established probability theory

**Goal:** Map net exposure to probability using mathematically rigorous approach.

**Parameters:**
- `exposure`: Net exposure value (positive = bullish, negative = bearish)

**Returns:** Probability ∈ [0.5-k, 0.5+k]

### 2. compute_mu_from_probability(P: float, S: float = None, K: float = None) → float

**Mathematical Approach:**
Direct application of Black-Scholes framework with enhanced scaling for wider drift range.

**Core Black-Scholes Relationship:**
In the Black-Scholes model, the probability of ending above strike K is:
```
P = Φ(d₂)
```
Where:
```
d₂ = (ln(S/K) + (μ - 0.5σ²)T) / (σ√T)
```

**Solving for μ:**
```
d₂ = Φ⁻¹(P)  # Inverse normal CDF
μ = [d₂ * σ * √T - ln(S/K)] / T + 0.5σ²
```

**Enhanced Scaling:**
```python
prob_distance = |P - 0.5| / 0.5  # Normalized distance from neutral
scaling_factor = 1.7  # Adjustment parameter for desired drift range

if base_mu > 0:
    scaled_mu = base_mu * (1 + scaling_factor * prob_distance)
else:
    scaled_mu = base_mu * (1 + scaling_factor * prob_distance)
```

**Goal:** Convert probability to drift parameter using established financial theory.

**Parameters:**
- `P`: Desired probability of ending above strike
- `S`: Current spot price (optional, defaults to S_0)
- `K`: Strike price (optional, defaults to S for "above current" probability)

**Returns:** Drift parameter μ with scaling to achieve range ±100

### 3. generate_gbm_path(mu: float, duration_in_seconds: int, random_seed: Optional[int] = None) → np.ndarray

**Mathematical Approach:**
Standard Geometric Brownian Motion (GBM) implementation following the stochastic differential equation.

**GBM Stochastic Differential Equation:**
```
dS = S * (μ dt + σ dW)
```
Where dW is a Wiener process (Brownian motion increment).

**Discrete Implementation:**
```
log_return = (μ - σ²/2) * dt + σ * √dt * Z
S[t+1] = S[t] * exp(log_return)
```
Where Z ~ N(0,1) is a standard normal random variable.

**Mathematical Properties:**
1. **Log-Normal Distribution**: Prices follow log-normal distribution
2. **Constant Parameters**: μ and σ remain constant throughout the path
3. **Martingale Property**: Under risk-neutral measure, discounted prices are martingales
4. **Path Continuity**: Continuous sample paths (no jumps)

**Goal:** Generate realistic price paths following established stochastic process.

**Parameters:**
- `mu`: Drift parameter (annualized)
- `duration_in_seconds`: Simulation length
- `random_seed`: For reproducibility

**Returns:** Array of price levels following GBM dynamics

### 4. generate_dynamic_exposure_path(exposure_series: pd.Series, ...) → Tuple

**Mathematical Approach:**
Three-pass algorithm with exponential weighted moving average smoothing.

**Pass 1 - Raw Drift Calculation:**
For each exposure value:
1. Map exposure → probability using `exposure_to_probability()`
2. Map probability → drift using `compute_mu_from_probability()`

**Pass 2 - Exponential Weighted Moving Average:**
```python
alpha = 0.1  # Smoothing factor
weights = [(1 - alpha)^j for j in range(window_size-1, -1, -1)]
weights = weights / sum(weights)  # Normalize to sum to 1
smoothed_drift[i] = sum(window_values * weights)
```

**Mathematical Properties of EWMA:**
1. **Exponential Decay**: Recent values have exponentially higher weights
2. **Stability**: Reduces noise while preserving trends
3. **Responsiveness**: Balances smoothness with adaptability
4. **Mathematical Foundation**: Well-established in time series analysis

**Pass 3 - GBM Path Generation:**
Standard GBM with time-varying drift:
```
log_return = (μ[t] - σ²/2) * dt + σ * √dt * N(0,1)
```

**Goal:** Create smoothed, stable price paths that respond to exposure changes.

**Parameters:**
- `exposure_series`: Time series of exposure values
- `random_seed`: For reproducibility
- `ma_window`: Window size for moving average (default: 12)

**Returns:** (price_path, drift_path, smoothed_drift_path, probability_path)

### 5. validate_statistics(price_path: np.ndarray, expected_mu: float) → Dict

**Mathematical Approach:**
Direction-based validation using realized statistics.

**Realized Statistics Calculation:**
1. **Log Returns**: `log_returns = diff(log(price_path))`
2. **Realized Volatility**: `σ_realized = std(log_returns) / √dt`
3. **Realized Drift**: `μ_realized = mean(log_returns) / dt + 0.5 * σ_realized²`
4. **Total Return**: `(S_final / S_initial) - 1`

**Validation Logic:**
```python
if expected_mu > 0:
    mu_valid = price_path[-1] > price_path[0]  # Final > Initial
elif expected_mu < 0:
    mu_valid = price_path[-1] < price_path[0]  # Final < Initial
else:  # expected_mu ≈ 0
    mu_valid = abs(total_return) < 0.01  # Within 1% of initial
```

**Goal:** Validate that simulated paths match expected drift direction.

**Parameters:**
- `price_path`: Simulated price array
- `expected_mu`: Expected drift parameter

**Returns:** Dictionary with validation results and realized statistics

## Key Mathematical Features

### 1. Normal CDF Probability Mapping
**Advantages:**
- **Theoretical Foundation**: Based on established probability theory
- **Smooth Transitions**: Continuously differentiable function
- **Symmetric Response**: Equal sensitivity to positive/negative exposure
- **Bounded Output**: Always produces valid probabilities

**Mathematical Properties:**
- **Domain**: (-∞, +∞) → [0, 1]
- **Monotonic**: Strictly increasing function
- **Inflection Point**: At x = 0 (neutral exposure)
- **Asymptotic**: Approaches 0 and 1 asymptotically

### 2. Black-Scholes Drift Calculation
**Theoretical Foundation:**
Based on the fundamental Black-Scholes relationship between probability and drift:
```
P(S_T > K) = Φ(d₂) where d₂ = (ln(S/K) + (μ - 0.5σ²)T) / (σ√T)
```

**Mathematical Rigor:**
- **Inverse Mapping**: Uses inverse normal CDF for exact calculation
- **Scaling Enhancement**: Applies consistent scaling for desired range
- **Parameter Consistency**: Maintains Black-Scholes assumptions

### 3. Exponential Weighted Moving Average
**Mathematical Formula:**
```
EWMA[t] = α * X[t] + (1-α) * EWMA[t-1]
```
Where α is the smoothing parameter.

**Properties:**
- **Exponential Decay**: Weights decrease exponentially with age
- **Stability**: Reduces high-frequency noise
- **Responsiveness**: Controlled by smoothing parameter α
- **Memory**: Incorporates all historical values with decreasing weights

### 4. Standard GBM Implementation
**Stochastic Process:**
```
dS/S = μ dt + σ dW
```

**Discrete Solution:**
```
S[t+1] = S[t] * exp((μ - σ²/2) * dt + σ * √dt * Z)
```

**Statistical Properties:**
- **Log-Normal**: Prices follow log-normal distribution
- **Constant Volatility**: σ remains constant
- **Independent Increments**: Non-overlapping increments are independent
- **Continuous Paths**: No discontinuous jumps

## Parameter Sensitivity Analysis

### High Impact Parameters:
- **scale**: Directly controls exposure sensitivity in probability mapping
- **k**: Constrains the probability range and affects drift magnitude
- **scaling_factor**: Controls the final drift range (±100 target)

### Medium Impact Parameters:
- **sigma**: Affects drift calculation through Black-Scholes formula
- **T**: Influences drift magnitude in Black-Scholes relationship
- **alpha**: Controls smoothing strength in EWMA

### Low Impact Parameters:
- **S_0**: Only affects initial price level, not dynamics
- **dt**: Affects simulation granularity but not core behavior

## Mathematical Advantages

1. **Theoretical Soundness**: All components based on established financial mathematics
2. **Predictable Behavior**: Well-understood mathematical properties
3. **Computational Efficiency**: Standard algorithms with known complexity
4. **Parameter Interpretability**: Clear mathematical meaning for all parameters
5. **Validation Simplicity**: Straightforward validation criteria
6. **Stability**: EWMA smoothing provides stable, noise-reduced behavior

## Limitations and Considerations

1. **Fixed Volatility**: No adaptive volatility modeling
2. **Symmetric Response**: Equal treatment of positive/negative exposure
3. **Linear Scaling**: Fixed scaling relationships throughout
4. **Smoothing Lag**: EWMA introduces lag in response to rapid changes
5. **No Market Psychology**: Purely mathematical approach without behavioral factors
6. **Constant Parameters**: GBM assumes constant drift and volatility

## Comparison with Advanced Approaches

### Strengths of Basic Approach:
- **Mathematical Transparency**: Clear, interpretable relationships
- **Computational Efficiency**: Lower computational requirements
- **Theoretical Foundation**: Based on proven financial mathematics
- **Stability**: Consistent, predictable behavior
- **Simplicity**: Easier to understand and implement

### Areas for Enhancement:
- **Market Psychology**: Could incorporate behavioral finance concepts
- **Adaptive Parameters**: Could implement time-varying parameters
- **Advanced Smoothing**: Could use more sophisticated smoothing techniques
- **Asymmetric Response**: Could differentiate between bullish/bearish scenarios
