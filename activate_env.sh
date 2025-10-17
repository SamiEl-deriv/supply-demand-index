#!/bin/bash
# Activation script for the supply-demand environment

echo "🚀 Activating supply-demand environment..."
source supply_demand_env/bin/activate

echo "✅ Environment activated!"
echo "📦 Python executable: $(which python)"
echo "📦 Python version: $(python --version)"
echo ""
echo "🔧 Available packages include:"
echo "   - numpy, pandas, scipy, scikit-learn"
echo "   - matplotlib, plotly, dash"
echo "   - yfinance, python-binance, pybit"
echo "   - fbm (fractional Brownian motion)"
echo "   - jupyter, ipython"
echo "   - and many more..."
echo ""
echo "💡 To run your main script: python 'RandD/deal cancellation/main.py'"
echo "💡 To deactivate: deactivate"
