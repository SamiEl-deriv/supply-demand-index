# IndexBuilder Package

## Overview

The IndexBuilder package provides tools for generating and managing various types of indices used in Deriv's trading systems. It supports both synthetic and financial indices, with capabilities for real-time data feeds and historical data analysis.

## Components

### 1. IndexBuilder Class

Base class for building synthetic indices with the following features:

- Index feed generation with configurable parameters
- Volatility and time-based calculations
- Statistical analysis tools
- Visualization capabilities using Plotly
- Support for multiple simulation runs

#### Key Parameters:
- `S0`: Initial index value
- `vol`: Annualized index volatility
- `T`: Time of simulation (in seconds)

### 2. FX Derived Module

Advanced module for creating derived FX indices with sophisticated features:

- Event-based index adjustments
- Multiple transformation methods (norm2t, t2norm, mixing)
- Intraday time splits handling
- Comprehensive statistical analysis
- Support for various shift types (log, arcsinh)

#### Key Features:
- Event detection and handling
- Lookback period calculations
- Volatility normalization
- Multi-period index construction

### 3. Financial Feed Module

Module for accessing historical market data:

- Direct connection to Deriv's feed servers
- Support for multiple financial instruments
- Efficient data retrieval and storage
- CSV export capabilities

#### Usage Example:
```python
from deriv_quant_package.indexbuilder import Feed

# Initialize feed
feed = Feed(
    start_date="2022-10-01",
    end_date="2022-10-25",
    symbol="frxUSDJPY"
)

# Get data with optional CSV save
data = feed.get_data(save_csv=True)
```

## Installation

The package is part of deriv_quant_package. No separate installation is required.

## Dependencies

- numpy: Numerical computations
- pandas: Data manipulation
- plotly: Visualization
- scipy: Statistical functions
- psycopg2: Database connectivity
- tqdm: Progress bars

## Important Notes

1. Server Access:
   - Financial feed access requires proper credentials
   - Must be run from a QA box with server access

2. Data Handling:
   - Large datasets are handled in chunks
   - Data is automatically saved in monthly tables
   - CSV export is optional but recommended for large datasets

3. Performance:
   - Index calculations are optimized for speed
   - Memory usage is managed through chunked processing
   - Parallel processing available for multi-period runs

## Security Note

When using the Financial Feed module:
- Always use secure credential management
- Access is restricted to authorized QA environments
- Do not share or expose connection credentials
