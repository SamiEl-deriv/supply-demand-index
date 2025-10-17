import numpy as np
import pandas as pd
from moments import MomentAnalyzer
from visualization import Visualizer


class IndexAnalysis:
    """Class for performing comprehensive risk and performance analysis."""

    def __init__(self):
        self._tracking_error = None
        self._information_ratio = None
        self._rolling_beta = None
        self._realized_volatility = None
        self._momentum_score = None
        self._trend_metrics = None
        self._autocorrelation = None
        
        # Initialize helper classes
        self.moment_analyzer = MomentAnalyzer()
        self.visualizer = Visualizer()

    def calculate_return_moments(self, returns, window=None, annualize=True, plot=False):
        """
        Calculate return distribution moments.

        Args:
            returns (pd.Series): Series of returns.
            window (int, optional): Rolling window size.
            annualize (bool): Whether to annualize the moments.
            plot (bool): Whether to plot rolling moments.

        Returns:
            dict: Dictionary containing moment calculations.
        """
        if window is None:
            moments = self.moment_analyzer.calculate_static_moments(returns, annualize)
        else:
            moments = self.moment_analyzer.calculate_rolling_moments(returns, window, annualize)
            if plot:
                self.visualizer.plot_moments(moments)
        return moments

    def calculate_tracking_error(self, returns, benchmark_returns):
        """Calculate tracking error vs benchmark."""
        tracking_diff = returns - benchmark_returns
        self._tracking_error = np.std(tracking_diff) * np.sqrt(252)  # Annualized
        return self._tracking_error

    def calculate_information_ratio(self, returns, benchmark_returns):
        """Calculate information ratio."""
        tracking_error = self.calculate_tracking_error(returns, benchmark_returns)
        active_return = (returns - benchmark_returns).mean() * 252  # Annualized
        self._information_ratio = active_return / tracking_error
        return self._information_ratio

    def calculate_rolling_beta(self, returns, benchmark_returns, window=60):
        """Calculate rolling beta."""
        rolling_cov = returns.rolling(window).cov(benchmark_returns)
        rolling_var = benchmark_returns.rolling(window).var()
        self._rolling_beta = rolling_cov / rolling_var
        return self._rolling_beta

    def calculate_realized_volatility(self, returns, window=21):
        """Calculate realized volatility."""
        self._realized_volatility = returns.rolling(window).std() * np.sqrt(252)
        return self._realized_volatility

    def calculate_momentum_score(self, returns, lookback_periods=[21, 63, 126, 252]):
        """Calculate momentum indicators."""
        cumulative_returns = (1 + returns).cumprod()
        momentum_scores = {}
        
        for period in lookback_periods:
            momentum = cumulative_returns / cumulative_returns.shift(period) - 1
            momentum_scores[f'{period}d_momentum'] = momentum.iloc[-1]
        
        self._momentum_score = momentum_scores
        return self._momentum_score

    def analyze_trend(self, returns, window=21):
        """Analyze trend strength and persistence."""
        prices = (1 + returns).cumprod()
        
        # Calculate moving averages
        ma_short = prices.rolling(window).mean()
        ma_long = prices.rolling(window * 2).mean()
        
        # Trend strength
        trend_strength = (prices - ma_long) / ma_long
        
        # Trend persistence (percentage of time price > MA)
        trend_persistence = (prices > ma_long).mean()
        
        self._trend_metrics = {
            'trend_strength': trend_strength.iloc[-1],
            'trend_persistence': trend_persistence,
            'current_trend': 'Uptrend' if prices.iloc[-1] > ma_long.iloc[-1] else 'Downtrend'
        }
        return self._trend_metrics

    def calculate_autocorrelation(self, returns, lags=20, plot=True):
        """Calculate and optionally plot return autocorrelations."""
        # Calculate autocorrelation
        acf = pd.Series([1] + [returns.autocorr(lag=i) for i in range(1, lags + 1)])
        
        # Calculate partial autocorrelation
        pacf = pd.Series()
        for i in range(lags + 1):
            if i == 0:
                pacf[i] = 1
            else:
                y = returns.values[i:]
                X = pd.DataFrame({f'lag_{j}': returns.shift(j).values[i:] for j in range(1, i + 1)})
                try:
                    model = np.linalg.lstsq(X, y, rcond=None)[0]
                    pacf[i] = model[-1]
                except:
                    pacf[i] = np.nan

        # Calculate confidence bands (95%)
        conf_level = 1.96 / np.sqrt(len(returns))
        
        self._autocorrelation = {
            'acf': acf,
            'pacf': pacf,
            'confidence_level': conf_level,
            'significant_lags': {
                'acf': [i for i, v in enumerate(acf) if abs(v) > conf_level],
                'pacf': [i for i, v in enumerate(pacf) if abs(v) > conf_level]
            }
        }

        if plot:
            self.visualizer.plot_autocorrelation(acf, pacf, conf_level)
        
        return self._autocorrelation

    def visualize_drawdown(self, returns):
        """Visualize cumulative returns and drawdown."""
        return self.visualizer.plot_drawdown(returns)

    def visualize_rolling_metrics(self, dates, rolling_beta, realized_vol):
        """Visualize rolling beta and realized volatility."""
        return self.visualizer.plot_rolling_metrics(dates, rolling_beta, realized_vol)
