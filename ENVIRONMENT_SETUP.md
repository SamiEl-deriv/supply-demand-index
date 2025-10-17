# Supply-Demand Environment Setup

## Overview
This project now has a dedicated Python virtual environment with all necessary packages installed to avoid kernel issues.

## Environment Details
- **Environment Name**: `supply_demand_env`
- **Python Version**: 3.13.2
- **Location**: `./supply_demand_env/`

## Installed Packages
### Core Scientific Computing
- numpy >= 2.2.0
- pandas >= 2.2.3
- scipy >= 1.14.1
- scikit-learn >= 1.6.0

### Visualization
- matplotlib >= 3.10.0
- plotly >= 6.0.0
- dash >= 2.18.2
- dash-bootstrap-components >= 1.7.1

### Financial Data & APIs
- yfinance >= 0.2.0
- python-binance >= 1.0.27
- pybit >= 5.9.0
- coinbase >= 2.1.0

### Specialized Libraries
- fbm >= 0.3.0 (Fractional Brownian Motion)
- ipykernel, ipython, jupyter (Interactive computing)

### And many more utilities...

## Usage

### For Python Scripts (Terminal)
```bash
# Activate environment with helper script
./activate_env.sh

# Or manually activate
source supply_demand_env/bin/activate

# Run your scripts
python "RandD/deal cancellation/main.py"
python test_imports.py

# Deactivate when done
deactivate
```

### For Jupyter Notebooks / VSCode Interactive
**IMPORTANT**: To use the environment in notebooks, you must select the correct kernel:

1. **In VSCode**: 
   - Open your notebook (.ipynb file)
   - Click on the kernel selector (top right, usually shows "Python 3.x.x")
   - Select "Supply Demand Environment" from the list
   - Now all packages (including fbm, yfinance, etc.) will be available

2. **In Jupyter Lab/Notebook**:
   - Open your notebook
   - Go to Kernel → Change Kernel → "Supply Demand Environment"
   - Or create a new notebook and select "Supply Demand Environment" kernel

3. **Test the kernel**:
   ```python
   # Run this in a notebook cell to verify
   from fbm import fbm
   import yfinance as yf
   print("✅ All packages available!")
   ```

### Installing Additional Packages
```bash
# Activate environment first
source supply_demand_env/bin/activate

# Install new packages
pip install package_name

# Update requirements.txt if needed
pip freeze > requirements.txt
```

## Files Created
- `requirements.txt` - Complete package list
- `activate_env.sh` - Helper activation script
- `supply_demand_env/` - Virtual environment directory
- `ENVIRONMENT_SETUP.md` - This documentation

## Troubleshooting
- If you encounter import errors, ensure the environment is activated
- The environment is isolated from your system Python to avoid conflicts
- All packages are pinned to specific versions for reproducibility

## Testing
The environment has been tested with:
- ✅ Basic imports (numpy, pandas, matplotlib)
- ✅ Financial modeling scripts
- ✅ Fractional Brownian Motion simulations
- ✅ All plotting functionality

No more kernel problems! 🎉
