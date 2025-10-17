# Adjusted Vol75 Supply-Demand Analysis Documentation

## Overview

This script analyzes Vol 75 trade data using the adjusted supply-demand index engine with advanced mathematical models including smooth non-linear exposure-to-probability mapping, adaptive drift calculation with market regime awareness, and sophisticated stochastic processes.

## Mathematical Framework

### Core Philosophy
- **High-Resolution Analysis**: Per-second interpolation for maximum responsiveness
- **Advanced Smoothing**: Weighted moving averages with long windows
- **Market Psychology Integration**: Incorporates behavioral finance concepts
- **Real-Time Responsiveness**: Immediate reaction to exposure changes

## Key Functions and Mathematical Approaches

### 1. load_vol75_trade_data(data_dir: str) → pd.DataFrame

**Mathematical Approach:**
Data aggregation and standardization for financial time series analysis.

**Process:**
1. **File Aggregation**: Combines multiple CSV files chronologically
2. **Timestamp Normalization**: Converts mixed datetime formats to standard pandas datetime
3. **Action Mapping**: Converts binary action codes to readable format
   - `0 → "buy"` (positive market pressure)
   - `1 → "sell"` (negative market pressure)
4. **Volume Standardization**: Uses `volume_usd` as the standard amount measure

**Goal:** Create a unified, chronologically ordered dataset for exposure analysis.

**Parameters:**
- `data_dir`: Directory path containing Vol 75 CSV files

**Returns:** Standardized DataFrame with columns: [timestamp, direction, amount]

### 2. calculate_net_exposure(trades_df: pd.DataFrame, window_minutes: int = 5) → pd.Series

**Mathematical Approach:**
Time-windowed exposure calculation using discrete time intervals.

**Formula:**
```
Net_Exposure[t] = Σ(Buy_Volume[t]) - Σ(Sell_Volume[t])
```
Where t represents each time window.

**Process:**
1. **Time Window Creation**: 
   ```
   windows = pd.date_range(start=start_time, end=end_time, freq=f"{window_minutes}min")
   ```
2. **Exposure Calculation**: For each window [t, t+Δt]:
   ```
   buys = Σ(volume | direction == "buy")
   sells = Σ(volume | direction == "sell")
   net_exposure = buys - sells
   ```
3. **Sparse Representation**: Only stores windows with non-zero exposure

**Goal:** Convert high-frequency trade data into exposure time series.

**Parameters:**
- `trades_df`: DataFrame with trade data
- `window_minutes`: Time window size (default: 5 minutes)

**Returns:** Time series of net exposure values

### 3. run_analysis(exposure_data: pd.Series, sample_size: int = 50, random_seed: int = 1505) → Tuple

**Mathematical Approach:**
Statistical sampling and supply-demand index generation using the adjusted engine.

**Engine Configuration:**
```python
engine = SupplyDemandIndexEngine(
    sigma=0.3,              # 30% volatility (conservative for crypto)
    scale=150_000,          # Higher sensitivity than basic version
    k=0.4,                  # Probability range [0.1, 0.9]
    T=1.0 / (365 * 24),     # 1-hour time horizon
    S_0=100_000,            # Starting price
    smoothness_factor=2.0   # Enhanced smoothness control
)
```

**Sampling Strategy:**
- **Random Sampling**: `np.random.choice()` for unbiased exposure selection
- **Sample Size Control**: Limits computational complexity while maintaining statistical validity

**Goal:** Generate supply-demand responsive index paths from Vol 75 exposure data.

**Parameters:**
- `exposure_data`: Time series of net exposure values
- `sample_size`: Number of exposure points to analyze (default: 50)
- `random_seed`: For reproducible results

**Returns:** (engine_instance, results_dictionary)

### 4. generate_dynamic_index_path(...) → None

**Mathematical Approach:**
High-resolution dynamic path generation with advanced interpolation and smoothing.

**Key Mathematical Components:**

#### 4.1 Interpolation to Per-Second Resolution
```python
# Original index (5-minute intervals)
original_index = np.arange(len(exposure_series))

# New index (per-second resolution)
seconds_per_minute = 60
new_index = np.linspace(0, len(exposure_series) - 1, 
                       len(exposure_series) * seconds_per_minute)

# Linear interpolation
interpolated_exposure = np.interp(new_index, original_index, exposure_series.values)
```

**Mathematical Justification:**
- **Nyquist Theorem**: Ensures no information loss during interpolation
- **Linear Interpolation**: Preserves monotonicity and bounded variation
- **Temporal Consistency**: Maintains causal relationships in time series

#### 4.2 Advanced Moving Average (900-point window)
```python
ma_window = 900  # 15 hours with per-second data
```

**Window Size Calculation:**
- 900 seconds = 15 minutes of per-second data
- Provides substantial smoothing while maintaining responsiveness
- Balances noise reduction with signal preservation

#### 4.3 Monte Carlo Simulation Framework
```python
for i in range(num_simulations):
    path, drift, smoothed_drift, prob = engine.generate_dynamic_exposure_path(
        exposure_series=interpolated_exposure_series,
        random_seed=random_seed + i,
        ma_window=ma_window
    )
```

**Statistical Properties:**
- **Independent Paths**: Each simulation uses different random seed
- **Confidence Intervals**: 10th and 90th percentiles for 80% confidence bands
- **Central Tendency**: Mean path represents expected behavior

**Goal:** Generate high-resolution, statistically robust index paths.

**Parameters:**
- `engine`: Initialized SupplyDemandIndexEngine
- `exposure_series`: Time series of exposure values
- `num_simulations`: Number of Monte Carlo paths (default: 100)
- `random_seed`: Base seed for reproducibility
- `ma_window`: Moving average window size (default: 900)
- `seconds_per_point`: Resolution control (default: 1 second)

### 5. create_combined_exposure_index_plot(...) → None

**Mathematical Approach:**
Dual-axis visualization with temporal alignment and scaling normalization.

**Temporal Alignment:**
```python
if len(index_path) != len(exposure):
    # Linear interpolation for length matching
    x_orig = np.linspace(0, 1, len(index_path))
    x_new = np.linspace(0, 1, len(exposure))
    index_path = np.interp(x_new, x_orig, index_path)
```

**Dual-Axis Scaling:**
- **Primary Axis**: Exposure values in USD
- **Secondary Axis**: Index values (price levels)
- **Independent Scaling**: Each axis optimized for its data range

**Goal:** Visualize correlation between exposure and index behavior.

### 6. create_separate_visualizations(...) → None

**Mathematical Approach:**
Comprehensive statistical visualization suite with theoretical curve fitting.

#### 6.1 Exposure-Probability Mapping Visualization
```python
# Theoretical curve generation
exposure_range = np.linspace(min(exposures) * 1.2, max(exposures) * 1.2, 1000)
theoretical_probs = [engine.exposure_to_probability(exp) for exp in exposure_range]
```

**Mathematical Properties:**
- **High Resolution**: 1000 points for smooth curve visualization
- **Extended Range**: 120% of data range for extrapolation visualization
- **Theoretical Foundation**: Uses actual engine mapping function

#### 6.2 Probability-Drift Mapping Visualization
```python
# Theoretical curve for probability-drift relationship
prob_range = np.linspace(0.1, 0.9, 1000)
theoretical_mu = [engine.compute_mu_from_probability(p) for p in prob_range]
```

**Statistical Validation:**
- **Scatter Plot**: Shows actual data points vs theoretical curve
- **Residual Analysis**: Visual assessment of model fit quality
- **Reference Lines**: Zero drift and neutral probability markers

#### 6.3 Price Path Visualization
**Monte Carlo Path Display:**
- **Sample Paths**: First 20 paths for individual trajectory visualization
- **Mean Path**: Central tendency in red for expected behavior
- **Reference Lines**: Initial price level for return calculation

#### 6.4 Validation Results Visualization
**Statistical Validation Metrics:**
- **Drift Validation Rate**: Percentage of paths matching expected direction
- **Bar Chart**: Clear visualization of validation success rates
- **Threshold Lines**: 50% reference for acceptable validation

**Goal:** Provide comprehensive visual analysis of all mathematical components.

## Analysis Configuration Parameters

### Engine Parameters (Adjusted Version)
```python
sigma = 0.3                    # 30% volatility (conservative)
scale = 150_000               # High exposure sensitivity
k = 0.4                       # Probability range [0.1, 0.9]
T = 1.0 / (365 * 24)         # 1-hour time horizon
S_0 = 100_000                 # Starting price
smoothness_factor = 2.0       # Enhanced smoothness
```

### Analysis Parameters
```python
window_minutes = 1            # 1-minute exposure windows
sample_size = 50             # Exposure points for detailed analysis
num_simulations = 20         # Monte Carlo paths
ma_window = 60               # 1-hour moving average (with 1-min data)
seconds_per_point = 1        # Per-second resolution
```

## Mathematical Advantages of Adjusted Approach

### 1. High-Resolution Processing
- **Per-Second Interpolation**: Maximum temporal resolution
- **Information Preservation**: No loss of exposure dynamics
- **Smooth Transitions**: Linear interpolation maintains continuity

### 2. Advanced Smoothing Strategy
- **Long Window**: 900-point moving average for stability
- **Weighted Averaging**: Exponential weighting for recent emphasis
- **Noise Reduction**: Substantial smoothing without losing signal

### 3. Enhanced Sensitivity
- **Higher Scale Parameter**: 150,000 vs 100,000 in basic version
- **Lower Volatility**: 30% vs 75% for more controlled behavior
- **Smoothness Factor**: Additional parameter for fine-tuning

### 4. Advanced Smoothing Strategy
- **Exponential Weighted Moving Average**: Same as basic version but with high-resolution data
- **Long Window**: 900-point moving average for substantial smoothing
- **Weighted Averaging**: Exponential weighting for recent emphasis
- **Noise Reduction**: Substantial smoothing without losing signal

### 5. Comprehensive Validation
- **Direction Validation**: Checks if drift direction matches price movement
- **Magnitude Validation**: Ensures reasonable return magnitudes
- **Statistical Robustness**: Multiple validation criteria

## Performance Characteristics

### Computational Complexity
- **Time Complexity**: O(n × m × s) where n=exposure points, m=simulations, s=path length
- **Space Complexity**: O(n × m) for storing all simulation paths
- **Interpolation Cost**: O(n × r) where r=resolution multiplier

### Statistical Properties
- **Convergence**: Monte Carlo paths converge to theoretical mean
- **Stability**: Long moving averages provide stable drift estimates
- **Responsiveness**: Per-second resolution captures rapid changes

### Memory Requirements
- **High-Resolution Data**: Significant memory for per-second interpolation
- **Multiple Simulations**: Storage for all Monte Carlo paths
- **Visualization Data**: Arrays for plotting and analysis

## Limitations and Considerations

1. **Computational Intensity**: High-resolution processing requires significant resources
2. **Memory Usage**: Per-second data and multiple simulations consume substantial memory
3. **Parameter Sensitivity**: Multiple parameters require careful calibration
4. **Interpolation Assumptions**: Linear interpolation may not capture all market dynamics
5. **Smoothing Trade-offs**: Long moving averages may lag rapid market changes
