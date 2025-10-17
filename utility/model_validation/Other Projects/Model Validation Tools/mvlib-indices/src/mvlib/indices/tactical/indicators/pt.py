import numpy as np
import pandas as pd
from .base import Signal, Strategy
from .signals import Discrete_Var, fill_zeros_with_last

class PT(Strategy):
    """
    Tactical Pair Trading Strategy Implementation using Bollinger Bands.
    
    This strategy:
    1. Calculates the ratio between two assets
    2. Applies Bollinger Bands to this ratio
    3. Generates trading signals based on the ratio's position relative to the bands
    
    Attributes
    ----------
    lookback : int
        Lookback period (s) for calculating rolling statistics.
    upper_threshold : float
        Number of standard deviations above mean for upper band.
    lower_threshold : float
        Number of standard deviations below mean for lower band.
    rebalancing : int
        Period of time (s) between each rebalancing. Defaults to 1.
    hold : bool
        Behaviour when ratio is between bands. Defaults to True.
        * True: hold the previous position
        * False: zero out the position
    signal_func : Signal
        The signal function, determining the position to take at each tick.

    Notes
    -----
    * Uses first tick each second for that second's signal
    * Uses fixed standard deviation multipliers for bands
    * Assumes the first asset in the pair is the numerator and second is denominator
    """

    num_of_underlyings = 2  # This strategy requires exactly 2 underlyings

    def __init__(self, lookback: int, thresholds: dict[str,float], hold_position=True, rebalancing: int=1, signal_func: str='discrete') -> None:
        """
        Parameters
        ----------
        lookback : int
            Lookback period (s).
        thresholds : dict[str,float]
            Dict containing 'upper_threshold' and 'lower_threshold' for standard deviation multipliers.
        hold_position : bool
            Behaviour when ratio is between bands. Defaults to True.
            * True: hold the previous position
            * False: zero out the position
        rebalancing : int
            Period of time (s) between each rebalancing. Defaults to 1.
        signal_func : str
            Type of signal function to use. Currently only 'discrete' is supported.
        """
        self.lookback = lookback
        self.upper_stdev = thresholds['upper_threshold']
        self.lower_stdev = thresholds['lower_threshold']
        self.rebalancing = rebalancing
        self.hold = hold_position
        
        # Initialize signal function
        self._SIGNALS = {'discrete': Discrete_Var}
        if signal_func not in self._SIGNALS:
            raise ValueError(f"signal_func must be one of {list(self._SIGNALS.keys())}")
        self.signal_func = self._SIGNALS[signal_func]()

    def get_signal(self, feed: pd.DataFrame) -> pd.DataFrame:
        """
        Gets the Pair Trading signals from a given feed containing two assets.

        Parameters
        ----------
        feed : pandas.DataFrame
            Financial feed containing spot prices for both assets.
            Expected columns: ['spot_1', 'spot_2'] for the two assets.
        
        Returns
        -------
        df_signal : pandas.DataFrame
            DataFrame containing:
            * ratio - Asset1/Asset2 ratio
            * sma - Simple Moving Average of the ratio
            * std - Standard deviation of the ratio
            * upper_band - Upper Bollinger Band
            * lower_band - Lower Bollinger Band
            * signal - Output of signal function
            * lookback - Number of ticks in lookback window
            * no_lb_period - Ticks without defined lookback window
        """
        # Ensure both assets are present
        required_cols = ['spot_1', 'spot_2']
        if not all(col in feed.columns for col in required_cols):
            raise ValueError(f"Feed must contain columns {required_cols}")

        # Calculate the ratio
        ratio = feed['spot_1'] / feed['spot_2']
        
        # Rebalanced index for signal calculation
        rebalancing = f'{self.rebalancing}s'
        rebalanced_index = feed.index.floor(rebalancing).drop_duplicates(keep='first')

        # Initialize series
        sma_series = pd.Series(np.nan, index=feed.index, name='sma')
        std_series = pd.Series(np.nan, index=feed.index, name='std')
        upper_band_series = pd.Series(np.nan, index=feed.index, name='upper_band')
        lower_band_series = pd.Series(np.nan, index=feed.index, name='lower_band')
        no_lb_period_series = pd.Series(None, index=feed.index, name='no_lb_period', dtype=bool)
        lookback_series = pd.Series(np.nan, index=feed.index, name='lookback')
        signal_series = pd.Series(np.nan, index=rebalanced_index, name='signal').asfreq(rebalancing)
        ratio_series = pd.Series(ratio, index=feed.index, name='ratio')

        # Process each trading day
        for name, group in feed.groupby(['trading_day']):
            # Skip non-trading times
            if name[0] == 0:
                continue

            # Get trading times for this day
            times = group.index
            
            # Calculate ratio for this group
            group_ratio = ratio[times]
            
            # Get Bollinger Bands components
            sma, std, lookback, no_lb_period = self.get_individual_stream(group_ratio)

            # Assign data to proper trading times
            sma_series[times] = sma
            std_series[times] = std
            upper_band_series[times] = sma + self.upper_stdev * std
            lower_band_series[times] = sma - self.lower_stdev * std
            no_lb_period_series[times] = no_lb_period
            lookback_series[times] = lookback
            ratio_series[times] = group_ratio

            # Cast ratio relative to bands to rebalancing period timeframe
            valid_ratio = group_ratio[~no_lb_period]
            bands_at_rebalancing = pd.DataFrame({
                'ratio': valid_ratio,
                'upper': upper_band_series[times][~no_lb_period],
                'lower': lower_band_series[times][~no_lb_period]
            })
            bands_at_rebalancing.index = bands_at_rebalancing.index.floor(rebalancing)
            bands_at_rebalancing = bands_at_rebalancing[~bands_at_rebalancing.index.duplicated(keep='first')]

            # Get signals based on ratio position relative to bands
            signal_piece = self.signal_func(
                bands_at_rebalancing['ratio'],
                bands_at_rebalancing['upper'],
                bands_at_rebalancing['lower']
            )
            
            # Hold previous position if enabled
            if self.hold:
                signal_piece = fill_zeros_with_last(signal_piece)
            signal_series[bands_at_rebalancing.index] = signal_piece

        # Combine all data
        pt_data = pd.merge_asof(
            ratio_series, signal_series,
            left_index=True, right_index=True,
            direction='backward'
        )
        pt_data = pd.concat([
            pt_data, sma_series, std_series, upper_band_series,
            lower_band_series, lookback_series, no_lb_period_series
        ], axis=1)
        
        return pt_data

    def get_individual_stream(self, ratio: pd.Series):
        """
        Calculates Bollinger Bands components for a ratio series.

        Parameters
        ----------
        ratio : pandas.Series
            Asset1/Asset2 ratio series for a single trading day.
        
        Returns
        -------
        sma : numpy.ndarray
            Simple Moving Average values.
        std : numpy.ndarray
            Standard deviation values.
        rolling_counts : numpy.ndarray
            Number of ticks in lookback window.
        no_lb_period : numpy.ndarray
            Ticks without defined lookback window.
        """
        # Find start of lookback period
        nearest_trading_start = ratio.index[0]
        
        # Identify periods without sufficient lookback
        no_lb_period = (ratio.index - nearest_trading_start) < pd.Timedelta(f'{self.lookback}s')

        # Calculate rolling statistics
        rolling_window = pd.Timedelta(seconds=self.lookback)
        rolling_counts = ratio.rolling(rolling_window).count().to_numpy()
        sma = ratio.rolling(rolling_window).mean().to_numpy()
        std = ratio.rolling(rolling_window).std().to_numpy()
        
        return sma, std, rolling_counts, no_lb_period
