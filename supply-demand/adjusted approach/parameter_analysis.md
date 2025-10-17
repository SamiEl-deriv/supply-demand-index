# Parameter Analysis for Dealing Team - Supply-Demand Index

## Executive Summary

This document identifies the most critical parameters that should be kept flexible for the dealing team to adjust in real-time when the supply-demand index behaves poorly. The analysis is based on the Complete Parameter Reference Table from the basic vs adjusted comparison.

## **High Priority Parameters - Most Critical for Real-Time Adjustment**

### 1. **`scale` (Exposure Normalization Factor)**
- **Current Values**: Basic: 100,000 | Adjusted: 150,000
- **Function**: Scales exposure values for probability mapping; higher = less sensitive
- **Why Critical**: Directly controls how sensitive the index is to exposure changes
- **When to Adjust**: 
  - **Increase** (e.g., to 200,000) if index is too reactive/volatile
  - **Decrease** (e.g., to 75,000) if index is not responding enough to market moves
- **Impact**: Higher = less sensitive, Lower = more sensitive
- **Recommended Range**: 50,000 - 300,000

### 2. **`k` (Maximum Probability Deviation)**
- **Current Values**: Basic: 0.45 | Adjusted: 0.4
- **Function**: Limits probability range to [0.5-k, 0.5+k]; lower = more conservative
- **Why Critical**: Controls the probability range, directly affecting drift magnitude
- **When to Adjust**:
  - **Decrease** (e.g., to 0.3) if index movements are too extreme
  - **Increase** (e.g., to 0.5) if index needs more dynamic range
- **Impact**: Lower = more conservative, Higher = more aggressive
- **Recommended Range**: 0.2 - 0.5

### 3. **`sigma` (Volatility Parameter)**
- **Current Values**: Basic: 0.75 (75%) | Adjusted: 0.3 (30%)
- **Function**: Controls price movement volatility; lower = smoother paths
- **Why Critical**: Controls price movement volatility and path smoothness
- **When to Adjust**:
  - **Decrease** if paths are too noisy/erratic
  - **Increase** if index needs more volatility to match market conditions
- **Impact**: Lower = smoother paths, Higher = more volatile paths
- **Recommended Range**: 0.1 - 1.0

## **Medium Priority Parameters - Important for Fine-Tuning**

### 4. **`alpha` (EWMA Smoothing Factor)**
- **Current Value**: 0.1 (both approaches)
- **Function**: Exponential decay rate; lower = more smoothing
- **Why Important**: Controls how much historical data influences current calculations
- **When to Adjust**:
  - **Decrease** (e.g., to 0.05) for more smoothing during volatile periods
  - **Increase** (e.g., to 0.2) for faster response to recent changes
- **Impact**: Lower = more smoothing/lag, Higher = more responsive
- **Recommended Range**: 0.02 - 0.3

### 5. **`ma_window` (Moving Average Window)**
- **Current Value**: 12 (both approaches)
- **Function**: Number of historical points for EWMA smoothing (1 hour)
- **Why Important**: Determines how many historical points are used for smoothing
- **When to Adjust**:
  - **Increase** (e.g., to 20) for more stability during choppy markets
  - **Decrease** (e.g., to 8) for faster response to trend changes
- **Impact**: Larger = more stable/lagged, Smaller = more responsive
- **Recommended Range**: 5 - 30

### 6. **`smoothness_factor` (Adjusted Only)**
- **Current Value**: 2.0
- **Function**: Controls sigmoid steepness in probability mapping
- **Why Important**: Controls the steepness of probability transitions
- **When to Adjust**:
  - **Increase** (e.g., to 3.0) for smoother, more gradual responses
  - **Decrease** (e.g., to 1.5) for sharper, more immediate responses
- **Impact**: Higher = smoother transitions, Lower = sharper responses
- **Recommended Range**: 1.0 - 4.0

## **Low Priority Parameters - Occasional Adjustment**

### 7. **`num_simulations` (Monte Carlo Count)**
- **Current Values**: Basic: 100 | Adjusted: 20
- **Function**: Trade-off between accuracy and computation time
- **Why Adjustable**: Balance between accuracy and processing speed
- **When to Adjust**: Increase for more accuracy, decrease for faster processing
- **Recommended Range**: 10 - 200

### 8. **`sample_size` (Analysis Points)**
- **Current Value**: 50 (both approaches)
- **Function**: Limits computational load while maintaining statistical validity
- **Why Adjustable**: Controls computational load vs statistical validity
- **When to Adjust**: Increase for more comprehensive analysis, decrease for speed
- **Recommended Range**: 20 - 100

## **Recommended Implementation for Dealing Team**

### **Real-Time Dashboard Controls**
```python
# High Priority - Always Accessible
scale_adjustment = slider(50_000, 300_000, current_value=150_000)
k_adjustment = slider(0.2, 0.5, current_value=0.4)
sigma_adjustment = slider(0.1, 1.0, current_value=0.3)

# Medium Priority - Advanced Controls
alpha_adjustment = slider(0.02, 0.3, current_value=0.1)
ma_window_adjustment = slider(5, 30, current_value=12)
smoothness_adjustment = slider(1.0, 4.0, current_value=2.0)  # Adjusted only
```

### **Quick Preset Configurations**

#### Conservative Mode (When Market is Too Volatile)
```python
conservative_params = {
    'scale': 200_000,      # Less sensitive to exposure changes
    'k': 0.3,             # More conservative probability range
    'sigma': 0.2,         # Lower volatility for smoother paths
    'alpha': 0.05,        # More smoothing, less noise
    'ma_window': 20,      # Longer smoothing window
    'smoothness_factor': 3.0  # Smoother transitions
}
```

#### Aggressive Mode (When Market is Too Slow)
```python
aggressive_params = {
    'scale': 75_000,       # More sensitive to exposure changes
    'k': 0.5,             # Wider probability range
    'sigma': 0.5,         # Higher volatility for more movement
    'alpha': 0.2,         # Less smoothing, more responsive
    'ma_window': 8,       # Shorter smoothing window
    'smoothness_factor': 1.5  # Sharper transitions
}
```

#### Balanced Mode (Standard Operation)
```python
balanced_params = {
    'scale': 150_000,      # Standard sensitivity
    'k': 0.4,             # Moderate probability range
    'sigma': 0.3,         # Standard volatility
    'alpha': 0.1,         # Standard smoothing
    'ma_window': 12,      # Standard smoothing window
    'smoothness_factor': 2.0  # Balanced transitions
}
```

## **Emergency Adjustment Guidelines**

### **Problem: Index is Too Volatile/Erratic**
**Solution**: Make index more conservative
- **Increase** `scale` (e.g., 150,000 → 200,000)
- **Decrease** `k` (e.g., 0.4 → 0.3)
- **Decrease** `sigma` (e.g., 0.3 → 0.2)
- **Decrease** `alpha` (e.g., 0.1 → 0.05)
- **Increase** `ma_window` (e.g., 12 → 20)

### **Problem: Index is Too Sluggish/Unresponsive**
**Solution**: Make index more aggressive
- **Decrease** `scale` (e.g., 150,000 → 100,000)
- **Increase** `k` (e.g., 0.4 → 0.45)
- **Increase** `alpha` (e.g., 0.1 → 0.15)
- **Decrease** `ma_window` (e.g., 12 → 8)

### **Problem: Index Lags Behind Market Movements**
**Solution**: Reduce smoothing and increase responsiveness
- **Decrease** `ma_window` (e.g., 12 → 8)
- **Increase** `alpha` (e.g., 0.1 → 0.15)
- **Decrease** `smoothness_factor` (e.g., 2.0 → 1.5)

### **Problem: Index is Too Noisy/Jumpy**
**Solution**: Increase smoothing
- **Increase** `ma_window` (e.g., 12 → 18)
- **Decrease** `alpha` (e.g., 0.1 → 0.07)
- **Increase** `smoothness_factor` (e.g., 2.0 → 2.5)

## **Parameter Interaction Matrix**

| Problem | scale | k | sigma | alpha | ma_window | smoothness_factor |
|---------|-------|---|-------|-------|-----------|-------------------|
| **Too Volatile** | ↑ | ↓ | ↓ | ↓ | ↑ | ↑ |
| **Too Sluggish** | ↓ | ↑ | - | ↑ | ↓ | ↓ |
| **Lagging Market** | - | - | - | ↑ | ↓ | ↓ |
| **Too Noisy** | - | - | ↓ | ↓ | ↑ | ↑ |

**Legend**: ↑ = Increase, ↓ = Decrease, - = No change needed

## **Monitoring and Validation**

### **Key Metrics to Watch After Parameter Changes**
1. **Validation Success Rate**: Should remain above 70%
2. **Exposure-Direction Correlation**: Should be positive and significant
3. **Price Path Smoothness**: Visual inspection of generated paths
4. **Response Time**: How quickly index reacts to exposure changes
5. **Volatility Match**: How well index volatility matches market conditions

### **Warning Signs of Poor Parameter Settings**
- **Validation success rate drops below 60%**
- **Index shows no correlation with exposure changes**
- **Paths become too smooth (no variation) or too erratic**
- **Index consistently over/under-reacts to market moves**
- **Computational performance degrades significantly**

## **Implementation Recommendations**

### **Phase 1: Basic Controls (Week 1-2)**
- Implement sliders for `scale`, `k`, and `sigma`
- Add Conservative/Balanced/Aggressive presets
- Monitor validation metrics

### **Phase 2: Advanced Controls (Week 3-4)**
- Add `alpha` and `ma_window` controls
- Implement emergency adjustment guidelines
- Add real-time performance monitoring

### **Phase 3: Full Implementation (Week 5-6)**
- Add `smoothness_factor` control (for Adjusted approach)
- Implement automated parameter suggestions based on market conditions
- Add historical parameter performance tracking

## **Risk Management**

### **Parameter Bounds (Hard Limits)**
```python
PARAMETER_BOUNDS = {
    'scale': (25_000, 500_000),      # Prevent extreme sensitivity
    'k': (0.1, 0.6),                # Maintain reasonable probability range
    'sigma': (0.05, 1.5),           # Prevent extreme volatility
    'alpha': (0.01, 0.5),           # Maintain smoothing effectiveness
    'ma_window': (3, 50),           # Reasonable smoothing window
    'smoothness_factor': (0.5, 5.0) # Reasonable transition control
}
```

### **Change Limits (Per Adjustment)**
- Maximum 20% change in any parameter per adjustment
- Require confirmation for changes > 50% from baseline
- Log all parameter changes with timestamps and reasons

## **Conclusion**

The three high-priority parameters (`scale`, `k`, `sigma`) provide the most direct control over index behavior and should be the primary focus for dealing team adjustments. The medium-priority parameters offer fine-tuning capabilities, while low-priority parameters can be adjusted occasionally for performance optimization.

**Key Success Factors:**
- Start with preset configurations before manual adjustments
- Monitor validation metrics after each change
- Document reasons for parameter changes
- Implement gradual changes rather than dramatic shifts
- Maintain parameter change logs for analysis and learning

This parameter analysis provides your dealing team with the tools and guidelines needed to maintain optimal index performance across varying market conditions.
