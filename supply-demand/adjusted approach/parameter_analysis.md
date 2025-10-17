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



This parameter analysis provides your dealing team with the tools and guidelines needed to maintain optimal index performance across varying market conditions.
