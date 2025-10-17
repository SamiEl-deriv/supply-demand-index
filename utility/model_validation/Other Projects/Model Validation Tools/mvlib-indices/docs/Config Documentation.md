# Configuration Documentation

This document details the parameter requirements for different strategy types for Tactical Index generation in the config.yaml file.

## Common Parameters

These parameters are common across all strategy types:

- `rebalancing` (integer): The rebalancing interval in seconds.
- `hold_position` (boolean): Whether to maintain positions between signals. If `False`, positions will always close in between signals.
- `strategy_type` (string): Type of strategy. Currently is one of `[RSI, MACO, BB, PT]`.
- `subtype` (string): Strategy subtype, either "trend" or "contrarian"
- `spread` (object): Spread configuration (For testing purposes, true spreads are set by Dealing team)
  - `alpha` (float): Spread alpha value, typically 1
  - `calibration` (float): Spread calibration value
  - `commission` (float): Commission value, typically 0
  - `signal_func` (string): Signal function type. Only `discrete` is functional. Added for forward compatibility, this entry is unnecessary until more signal types are developed

## RSI (Relative Strength Index) Strategy
RSI strategies measure momentum by comparing the magnitude of recent gains to recent losses.

### Required Parameters
- `lookback` (integer): Primary lookback period in seconds
- `secondary_lookback` (integer): Secondary lookback period in seconds
- `thresholds` (object): Signal thresholds
  - `lower_threshold` (float): Lower RSI threshold (typically less than 50)
  - `upper_threshold` (float): Upper RSI threshold (typically more than 59)
- `leverages` (object): Position sizing
  - `lower_leverage` (float): Leverage when RSI is below lower threshold
  - `upper_leverage` (float): Leverage when RSI is above upper threshold


### Example
```yaml
RSIXAGCL:
  lookback: 5400
  secondary_lookback: 3600
  thresholds:
    lower_threshold: 49.5
    upper_threshold: 51
  leverages:
    lower_leverage: 3
    upper_leverage: -1
  rebalancing: 1
  hold_position: True
  strategy_type: RSI
  subtype: contrarian
  signal_func: discrete
  spread:
    alpha: 1
    calibration: 0.006866
    commission: 0
```

## MACO (Moving Average Crossover) Strategy
MACO strategies generate signals based on the crossover of two moving averages. Currently, `subtype` is always trend here, unless new specs determine otherwise.

### Required Parameters
- `lookback` (integer): Slow moving average period in seconds
- `fast_lookback` (integer): Fast moving average period in seconds
- `leverages` (object): Position sizing
  - `lower_leverage` (float): Leverage for the slow MA crossing under the fast MA
  - `upper_leverage` (float): Leverage for the slow MA crossing above the fast MA

### Example
```yaml
MACOXAG1:
  lookback: 3600
  fast_lookback: 600
  leverages:
    lower_leverage: 1
    upper_leverage: 2
  rebalancing: 1
  hold_position: True
  strategy_type: MACO
  subtype: trend
  spread:
    alpha: 1
    calibration: 0.006866
    commission: 0
```

## BB (Bollinger Bands) Strategy
BB strategies use standard deviations from a moving average to identify overbought/oversold conditions.

### Required Parameters
- `lookback` (integer): Period for calculating moving average and standard deviation
- `thresholds` (object): Signal thresholds
  - `lower_threshold` (float): Lower band threshold (in standard deviations)
  - `upper_threshold` (float): Upper band threshold (in standard deviations)
- `leverages` (object): Position sizing
  - `lower_leverage` (float): Leverage when price crosses lower band
  - `upper_leverage` (float): Leverage when price crosses upper band

### Example
```yaml
BBXAUCL:
  lookback: 3600
  rebalancing: 1
  thresholds:
    lower_threshold: 3.29
    upper_threshold: 3.48
  leverages:
    lower_leverage: 2
    upper_leverage: 1
  hold_position: True
  strategy_type: BB
  subtype: contrarian
  signal_func: discrete
  spread:
    alpha: 1
    calibration: 0
    commission: 0
```

## PT (Pairs Trading) Strategy
PT strategies trade the relative value between two correlated assets.

### Required Parameters
- `lookback` (integer): Period for calculating pair statistics
- `thresholds` (object): Signal thresholds
  - `lower_threshold` (float): Lower threshold for pair ratio
  - `upper_threshold` (float): Upper threshold for pair ratio
- `leverages` (object): Position sizing
  - `lower_leverage` (float): Leverage when ratio is below lower threshold
  - `upper_leverage` (float): Leverage when ratio is above upper threshold

### Example
```yaml
PTBTCETH1:
  lookback: 60
  rebalancing: 10
  thresholds:
    lower_threshold: 1
    upper_threshold: 4
  leverages:
    upper_leverage: 0.5
    lower_leverage: 9
  hold_position: True
  strategy_type: PT
  subtype: trend
  signal_func: discrete
  spread:
    alpha: 1
    calibration: 0
    commission: 0
```

## Strategy Naming Convention

The strategy names follow a pattern that indicates:
1. Strategy type (RSI, MACO, BB, PT)
2. Asset (pair) (XAG, XAU, EURUSD, USDJPY, GBPUSD, BTC, ETH)
3. Strategy variant:
   - CL/CS: Contrarian Large/Small
   - ML/MS: Momentum (Trend) Large/Small
   - Numbers (1,2): Different parameter sets for same strategy type

For example:
- RSIXAGCL: RSI strategy on Silver (XAG), Contrarian Large variant
- MACOXAU1: Moving Average Crossover strategy on Gold (XAU), parameter set 1
- BBXAUML: Bollinger Bands strategy on Gold (XAU), Momentum Large variant
- PTBTCETH1: Pairs Trading strategy on BTC/ETH pair, parameter set 1
