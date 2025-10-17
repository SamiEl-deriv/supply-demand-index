# RiskAnalysis Module

## Overview

The RiskAnalysis module provides comprehensive tools for analyzing financial time series data, with a focus on risk metrics, statistical analysis, and visualization. It consists of three main components that work together to provide a complete analysis toolkit.

## Components

### 1. IndexAnalysis

The main class that orchestrates risk and performance analysis. Key features:

- Return distribution analysis
- Tracking error and information ratio calculations
- Rolling beta analysis
- Realized volatility calculations
- Momentum scoring
- Trend analysis
- Autocorrelation analysis
- Drawdown visualization

### 2. MomentAnalyzer

Specialized class for statistical moment calculations:

- Static and rolling moment calculations
- Mean, variance, skewness, and kurtosis
- Jarque-Bera normality testing
- Moment convergence analysis
- Annualization options

### 3. Visualizer

Handles all visualization needs with consistent styling:

- Rolling moments plots
- Drawdown visualization
- Autocorrelation function plots
- Rolling metrics visualization

## Usage Examples

```python
from deriv_quant_package.RiskAnalysis import IndexAnalysis

# Initialize analyzer
analyzer = IndexAnalysis()

# Calculate and visualize return moments
moments = analyzer.calculate_return_moments(returns, window=60, annualize=True, plot=True)

# Analyze tracking error and information ratio
tracking_error = analyzer.calculate_tracking_error(returns, benchmark_returns)
info_ratio = analyzer.calculate_information_ratio(returns, benchmark_returns)

# Analyze trends and momentum
trend_metrics = analyzer.analyze_trend(returns, window=21)
momentum = analyzer.calculate_momentum_score(returns)

# Visualize drawdowns
analyzer.visualize_drawdown(returns)
```

## Dependencies

- numpy: Numerical computations
- pandas: Time series handling
- scipy.stats: Statistical calculations
- matplotlib: Visualization

## Installation

The module is part of the deriv_quant_package and is automatically installed with it. No separate installation is required.

## Notes

- All time series inputs should be pandas Series with datetime index
- Default annualization factor is 252 (trading days)
- Visualization settings are preconfigured for consistency but can be customized
- Rolling calculations automatically handle NaN values at the start of the series
