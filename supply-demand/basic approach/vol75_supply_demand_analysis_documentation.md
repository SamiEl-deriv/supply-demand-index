# Basic Vol75 Supply-Demand Analysis Documentation

## Overview

This script analyzes Vol 75 trade data using the basic supply-demand index engine with mathematically rigorous approaches including normal CDF-based exposure-to-probability mapping, Black-Scholes framework for drift calculation, and standard GBM price path generation with exponential weighted moving average smoothing.

## Mathematical Framework

### Core Philosophy
- **Mathematical Rigor**: Uses established financial mathematics and statistical theory
- **Standard Resolution**: 5-minute time windows for stable analysis
- **Traditional Smoothing**: Exponential weighted moving averages
- **Theoretical Foundation**: Based on proven Black-Scholes and normal distribution theory

## Key Functions and Mathematical Approaches

### 1. load_vol75_trade_data(data_dir: str) → pd.DataFrame

**Mathematical Approach:**
Standard data aggregation and normalization for financial time series analysis.

**Process:**
1. **File Aggregation**: Sequential loading and concatenation of CSV files
2. **Timestamp Standardization**: Converts mixed datetime formats using pandas `format="mixed"`
3. **Action Mapping**: Binary to categorical conversion
   - `0 → "buy"` (positive market pressure)
   - `1 → "sell"` (negative market pressure)
4. **Volume Normalization**: Uses `volume_usd` as standard measure

**Goal:** Create a standardized, chronologically ordered dataset for exposure analysis.

**Parameters:**
- `data_dir`: Directory path containing Vol 75 CSV files

**Returns:** Standardized DataFrame with columns: [timestamp, direction, amount]

### 2. calculate_net_exposure(trades_df: pd.DataFrame, window_minutes: int = 5) → pd.Series

**Mathematical Approach:**
Discrete time-windowed exposure calculation with fixed intervals.

**Mathematical Formula:**
```
Net_Exposure[t] = Σ(Buy_Volume[t]) - Σ(Sell_Volume[t])
```
For each time window t ∈ [t₀, t₀ + Δt] where Δt = window_minutes.

**Algorithm:**
1. **Time Grid Creation**: 
   ```
   windows = pd.date_range(start=start_time, end=end_time, freq=f"{window_minutes}min")
   ```
2. **Window-wise Aggregation**: For each window [tᵢ, tᵢ₊₁]:
   ```python
   mask = (timestamp >= tᵢ) & (timestamp < tᵢ₊₁)
   buys = sum(amount[mask & (direction == "buy")])
   sells = sum(amount[mask & (direction == "sell")])
   net_exposure = buys - sells
   ```
3. **Sparse Storage**: Only non-zero exposures are retained

**Goal:** Transform high-frequency trade data into manageable exposure time series.

**Parameters:**
- `trades_df`: DataFrame containing trade data
- `window_minutes`: Time window size (default: 5 minutes)

**Returns:** Time-indexed series of net exposure values

### 3. run_analysis(exposure_data: pd.Series, sample_size: int = 50, random_seed: int = 1505) → Tuple

**Mathematical Approach:**
Statistical sampling and index generation using the basic engine with Vol 75-optimized parameters.

**Engine Configuration:**
```python
engine = SupplyDemandIndexEngine(
    sigma=0.75,             # 75% volatility (matches Vol 75 characteristics)
    scale=100_000,          # Standard exposure sensitivity
    k=0.45,                 # Probability range [0.05, 0.95]
    T=1.0 / (365 * 24),     # 1-hour time horizon
    S_0=100_000,            # Starting price
    dt=1 / (86_400 * 365)   # 1-second time step
)
```

**Parameter Justification:**
- **σ = 0.75**: Matches Vol 75 index volatility characteristics
- **scale = 100,000**: Standard sensitivity for exposure mapping
- **k = 0.45**: Allows probability range [0.05, 0.95] for wide drift range
- **T = 1 hour**: Reasonable time horizon for probability calculations

**Sampling Strategy:**
```python
np.random.seed(random_seed)
sampled_indices = np.random.choice(len(exposure_data), size=sample_size, replace=False)
sampled_exposure = exposure_data.iloc[sampled_indices].values
```

**Goal:** Generate supply-demand responsive index paths using established mathematical methods.

**Parameters:**
- `exposure_data`: Time series of net exposure values
- `sample_size`: Number of exposure points to analyze (default: 50)
- `random_seed`: For reproducible results (default: 1505)

**Returns:** (engine_instance, results_dictionary)

### 4. generate_dynamic_index_path(...) → None

**Mathematical Approach:**
Standard resolution dynamic path generation with exponential weighted moving average smoothing.

**Key Mathematical Components:**

#### 4.1 Standard Resolution Processing
```python
seconds_per_point = 300  # 5-minute intervals
ma_window = 12          # 1 hour with 5-minute data
```

**Resolution Characteristics:**
- **Time Step**: 300 seconds (5 minutes) per data point
- **Window Coverage**: 12 points = 1 hour of data
- **Computational Efficiency**: Lower resolution reduces computational load
- **Stability**: Longer intervals provide more stable exposure estimates

#### 4.2 Exponential Weighted Moving Average
The basic engine uses EWMA smoothing in `generate_dynamic_exposure_path()`:

```python
alpha = 0.1  # Smoothing factor
weights = [(1 - alpha)^j for j in range(window_size-1, -1, -1)]
weights = weights / sum(weights)  # Normalize
smoothed_drift[i] = sum(window_values * weights)
```

**Mathematical Properties:**
- **Exponential Decay**: Recent values weighted more heavily
- **Smoothing Parameter**: α = 0.1 provides moderate smoothing
- **Stability**: Reduces noise in drift calculations
- **Memory**: Incorporates historical information with decreasing weights

#### 4.3 Monte Carlo Framework
```python
for i in range(num_simulations):
    path, drift, smoothed_drift, prob = engine.generate_dynamic_exposure_path(
        exposure_series=exposure_series,
        random_seed=random_seed + i,
        ma_window=ma_window
    )
```

**Statistical Properties:**
- **Independent Simulations**: Each path uses different random seed
- **Confidence Bands**: 10th and 90th percentiles for 80% confidence
- **Central Tendency**: Mean path represents expected behavior
- **Convergence**: Large number of simulations ensures statistical reliability

**Goal:** Generate stable, smoothed index paths with traditional time resolution.

**Parameters:**
- `engine`: Initialized SupplyDemandIndexEngine
- `exposure_series`: Time series of exposure values
- `num_simulations`: Number of Monte Carlo paths (default: 100)
- `random_seed`: Base seed for reproducibility (default: 1505)
- `ma_window`: Moving average window size (default: 12)
- `seconds_per_point`: Time resolution (default: 300 seconds = 5 minutes)

### 5. create_combined_exposure_index_plot(...) → None

**Mathematical Approach:**
Dual-axis visualization with temporal synchronization.

**Temporal Alignment Algorithm:**
```python
if len(index_path) != len(exposure):
    # Linear interpolation for length matching
    x_orig = np.linspace(0, 1, len(index_path))
    x_new = np.linspace(0, 1, len(exposure))
    index_path = np.interp(x_new, x_orig, index_path)
```

**Mathematical Properties:**
- **Linear Interpolation**: Preserves monotonicity and continuity
- **Temporal Consistency**: Maintains time-series relationships
- **Scaling Independence**: Each axis optimized for its data range

**Goal:** Visualize relationship between exposure and index behavior.

### 6. create_separate_visualizations(...) → None

**Mathematical Approach:**
Comprehensive visualization suite with theoretical curve validation.

#### 6.1 Exposure-Probability Mapping (Normal CDF)
```python
exposure_range = np.linspace(min(exposures) * 1.2, max(exposures) * 1.2, 1000)
theoretical_probs = [engine.exposure_to_probability(exp) for exp in exposure_range]
```

**Theoretical Curve Properties:**
- **Normal CDF**: Φ(x) = ∫_{-∞}^x (1/√(2π)) * e^(-t²/2) dt
- **Smooth S-Curve**: Continuously differentiable
- **Symmetric**: Equal response to positive/negative exposure
- **Bounded**: Output ∈ [0, 1] before scaling

#### 6.2 Probability-Drift Mapping (Black-Scholes)
```python
prob_range = np.linspace(0.1, 0.9, 1000)
theoretical_mu = [engine.compute_mu_from_probability(p) for p in prob_range]
```

**Black-Scholes Relationship:**
```
μ = [Φ⁻¹(P) * σ * √T - ln(S/K)] / T + 0.5σ²
```

**Mathematical Validation:**
- **Theoretical Foundation**: Based on established option pricing theory
- **Inverse Mapping**: Uses inverse normal CDF for exact calculation
- **Scaling Consistency**: Applied uniformly across probability range

#### 6.3 GBM Price Path Visualization
**Standard GBM Properties:**
- **Log-Normal Distribution**: Price levels follow log-normal distribution
- **Constant Parameters**: Fixed μ and σ within each path
- **Independent Increments**: Non-overlapping returns are independent
- **Continuous Paths**: No discontinuous jumps

#### 6.4 Validation Results
**Direction-Based Validation:**
```python
if expected_mu > 0:
    mu_valid = price_path[-1] > price_path[0]
elif expected_mu < 0:
    mu_valid = price_path[-1] < price_path[0]
else:
    mu_valid = abs(total_return) < 0.01
```

**Goal:** Provide comprehensive visual validation of mathematical components.

## Analysis Configuration Parameters

### Engine Parameters (Basic Version)
```python
sigma = 0.75                  # 75% volatility (Vol 75 characteristic)
scale = 100_000              # Standard exposure sensitivity
k = 0.45                     # Probability range [0.05, 0.95]
T = 1.0 / (365 * 24)         # 1-hour time horizon
S_0 = 100_000                # Starting price
dt = 1 / (86_400 * 365)      # 1-second time step
```

### Analysis Parameters
```python
window_minutes = 5           # 5-minute exposure windows
sample_size = 50            # Exposure points for detailed analysis
num_simulations = 100       # Monte Carlo paths
ma_window = 12              # 1-hour moving average (with 5-min data)
seconds_per_point = 300     # 5-minute resolution
```

## Mathematical Advantages of Basic Approach

### 1. Theoretical Rigor
- **Normal CDF**: Established probability theory foundation
- **Black-Scholes**: Proven option pricing framework
- **Standard GBM**: Well-understood stochastic process
- **EWMA**: Traditional time series smoothing

### 2. Computational Efficiency
- **Standard Resolution**: 5-minute intervals reduce computational load
- **Fixed Parameters**: No adaptive calculations required
- **Simple Algorithms**: Standard mathematical operations
- **Memory Efficiency**: Lower resolution data storage

### 3. Predictable Behavior
- **Symmetric Response**: Equal treatment of positive/negative exposure
- **Consistent Scaling**: Fixed relationships throughout
- **Stable Smoothing**: EWMA provides predictable noise reduction
- **Mathematical Transparency**: Clear interpretability

### 4. Parameter Interpretability
- **σ = 0.75**: Directly matches Vol 75 volatility
- **scale = 100,000**: Clear exposure sensitivity threshold
- **k = 0.45**: Explicit probability range constraint
- **α = 0.1**: Standard EWMA smoothing parameter

## Performance Characteristics

### Computational Complexity
- **Time Complexity**: O(n × m) where n=exposure points, m=simulations
- **Space Complexity**: O(n × m) for storing simulation results
- **Processing Speed**: Fast due to standard resolution and simple algorithms

### Statistical Properties
- **Convergence**: Monte Carlo paths converge to theoretical expectations
- **Stability**: EWMA smoothing provides stable drift estimates
- **Consistency**: Reproducible results with fixed random seeds

### Memory Requirements
- **Standard Resolution**: Moderate memory usage for 5-minute data
- **Simulation Storage**: Reasonable memory for 100 Monte Carlo paths
- **Visualization Data**: Standard arrays for plotting

## Limitations and Considerations

1. **Fixed Volatility**: No adaptive volatility modeling
2. **Symmetric Response**: Equal treatment of bullish/bearish exposure
3. **Standard Resolution**: 5-minute intervals may miss rapid changes
4. **Smoothing Lag**: EWMA introduces delay in response to exposure changes
5. **No Market Psychology**: Purely mathematical approach
6. **Linear Scaling**: Fixed scaling relationships throughout

## Comparison with Advanced Approaches

### Strengths of Basic Approach:
- **Mathematical Clarity**: Transparent, interpretable relationships
- **Computational Speed**: Efficient processing with standard algorithms
- **Theoretical Foundation**: Based on established financial mathematics
- **Stability**: Consistent, predictable behavior
- **Simplicity**: Easy to understand and validate

### Areas Where Advanced Approaches Excel:
- **Market Psychology**: Behavioral finance integration
- **Adaptive Parameters**: Time-varying sensitivity
- **High Resolution**: Per-second data processing
- **Advanced Smoothing**: Sophisticated filtering techniques
- **Asymmetric Response**: Different treatment of bullish/bearish scenarios

## Mathematical Validation

### Theoretical Consistency
- **Normal CDF Properties**: Monotonic, bounded, smooth
- **Black-Scholes Validity**: Consistent with option pricing theory
- **GBM Properties**: Log-normal distribution, independent increments
- **EWMA Convergence**: Exponential decay of historical influence

### Empirical Validation
- **Direction Validation**: Checks if drift direction matches price movement
- **Statistical Consistency**: Realized statistics match theoretical expectations
- **Parameter Sensitivity**: Predictable response to parameter changes
- **Convergence Testing**: Monte Carlo results converge with sufficient simulations
