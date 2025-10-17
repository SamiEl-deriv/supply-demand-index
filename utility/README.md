## Introduction

The deriv_quant_package is formed of six sub-packages:
- indexbuilder: Tools for building and managing indices
- pricer: Options pricing and calculations
- RiskAnalysis: Risk assessment and data analysis tools
- signals: Signal processing and analysis
- tests: Test suites and utilities
- tradingstrategies: Implementation of trading strategies

A description of each module (i.e the classes and functions they define) can be found in the README.md of each sub-package.

## Project Structure

```
deriv_quant_package/
├── indexbuilder/     # Index building tools
├── pricer/          # Options pricing
├── RiskAnalysis/    # Risk assessment tools
├── signals/         # Signal processing
├── tests/           # Test suites
├── tradingstrategies/ # Trading strategies
└── notebooks/       # Jupyter notebooks
    └── RiskAnalysis/  # Risk analysis notebooks
```

## How to use

Even though there are dependencies between some sub-packages, each of them can be loaded in a python script using one of the following statements:
```python
from deriv_quant_package.pricer import * 
```
or 
```python
import deriv_quant_package.indexbuilder 
```
Examples are given here for _pricer_ or _indexbuilder_ but could be applied to any sub-package.

## Notebooks

Jupyter notebooks are organized in the `notebooks` directory, grouped by their related sub-package. These notebooks contain:
- Examples and tutorials
- Research and analysis
- Development experiments

Each notebook should be self-contained and include proper documentation of its purpose and usage.
