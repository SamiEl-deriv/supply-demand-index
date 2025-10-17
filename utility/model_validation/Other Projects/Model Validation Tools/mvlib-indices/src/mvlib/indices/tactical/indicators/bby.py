import numpy as np
import numba as nb
import pandas as pd
from .base import Signal, Strategy
from .signals import Discrete_Var, fill_zeros_with_last

class BBY(Strategy):
    """
    Tactical Bollinger Bands Strategy Implementation.
    
    Attributes
    ----------
    lookback : int
        Lookback period (s).
    upper_threshold : float
        Number of standard deviations above mean for upper band.
    lower_threshold : float
        Number of standard deviations below mean for lower band.
    rebalancing : int
        Period of time (s) between each rebalancing, i.e observation of band crossings for signal calculation. Defaults to 1.
    hold : bool
        Behaviour when price is between bands. Defaults to True.
        * Enabling `hold_position` is equivalent to holding the previous position;
        * Disabling this is equivalent to zeroing out the position.
    signal_func : Signal
        The signal function, determining the position to take at each tick.

    Notes
    -----
    * Uses first tick each second for that second's signal;
    * Uses fixed standard deviation multipliers for bands.
    """

    num_of_underlyings = 1

    def __init__(self, lookback : int, thresholds : dict[str,float], 
                 hold_position = True, rebalancing : int = 1, signal_func : str = 'discrete') -> None:
        """
        Parameters
        ----------
        lookback : int
            Lookback period (s).
        secondary_lookback : int
            Secondary lookback period (s).
        thresholds : list[float]
            List of standard deviation multipliers: [upper_threshold, lower_threshold].
        hold_position : bool
            Behaviour when price is between bands. Defaults to True.
            * Enabling `hold_position` is equivalent to holding the previous position;
            * Disabling this is equivalent to zeroing out the position.
        rebalancing : int
            Period of time (s) between each rebalancing, i.e observation of band crossings for signal calculation. Defaults to 1.
        """
        self.lookback = lookback
        self.upper_stdev = thresholds['upper_threshold']
        self.lower_stdev = thresholds['lower_threshold']
        self.rebalancing = rebalancing
        self.hold = hold_position
        # Initialize signal function
        self.signal_func = _SIGNALS[signal_func]()

    def get_signal(self, feed : pd.DataFrame) -> pd.DataFrame:
        """
        Gets the Bollinger Bands signals from a given feed.

        Parameters
        ----------
        feed : pandas.DataFrame
            Financial feed processed with `feed.feed_processor`.
        
        Returns
        -------
        df_signal : pandas.DataFrame
            Pandas DataFrame containing the following columns:
            * `sma` - Simple Moving Average value;
            * `std` - Standard deviation value;
            * `upper_band` - Upper Bollinger Band;
            * `lower_band` - Lower Bollinger Band;
            * `signal` - Output of signal function;
            * `lookback` - The number of ticks in the lookback window;
            * `no_lb_period` - Ticks without a defined lookback window;
        """
        # Rebalanced index behaviour may break for durations not a factor of 60.
        rebalancing = f'{self.rebalancing}s'
        rebalanced_index = feed.index.floor(rebalancing).drop_duplicates(keep='first')

        # Define all required series
        sma_series = pd.Series(np.nan, index=feed.index, name='sma')
        std_series = pd.Series(np.nan, index=feed.index, name='std')
        upper_band_series = pd.Series(np.nan, index=feed.index, name='upper_band')
        lower_band_series = pd.Series(np.nan, index=feed.index, name='lower_band')
        no_lb_period_series = pd.Series(None, index=feed.index, name='no_lb_period', dtype=bool)
        lookback_series = pd.Series(np.nan, index=feed.index, name='lookback')
        signal_series = pd.Series(np.nan, index=rebalanced_index, name='signal').asfreq(rebalancing)
        
        # Filter to process on each trading day
        for name, group in feed.groupby(['trading_day']):
            # Remove non-trading times
            if name[0] == 0:
                continue

            # Retrieve trading times per day
            times = group.index
            # Get BBands & additional data per stream
            sma, std, lookback, no_lb_period = self.get_individual_stream(group['spot'])

            # Assign obtained data to proper trading times 
            sma_series[times] = sma
            std_series[times] = std
            upper_band_series[times] = sma + self.upper_stdev * std
            lower_band_series[times] = sma - self.lower_stdev * std
            no_lb_period_series[times] = no_lb_period
            lookback_series[times] = lookback

            # Cast price relative to bands to the rebalancing period time-frame
            price = pd.Series(group['spot'].values, index=times)[~no_lb_period]
            bands_at_rebalancing = pd.DataFrame({
                'price' : price,
                'upper' : upper_band_series[times][~no_lb_period],
                'lower' : lower_band_series[times][~no_lb_period]
            })
            bands_at_rebalancing.index = bands_at_rebalancing.index.floor(rebalancing)
            bands_at_rebalancing = bands_at_rebalancing[~bands_at_rebalancing.index.duplicated(keep='first')]

            # Get portfolio signals based on price position relative to bands
            signal_piece = self.signal_func(bands_at_rebalancing['price'], 
                                            bands_at_rebalancing['upper'], 
                                            bands_at_rebalancing['lower']
                                            )
            # Where the position would normally be closed, it preserves the position instead
            if self.hold:
                signal_piece = fill_zeros_with_last(signal_piece)
            signal_series[bands_at_rebalancing.index] = signal_piece

        # Combine data
        bby_data = pd.merge_asof(sma_series, signal_series, left_index=True, right_index=True, direction='backward')
        bby_data = pd.concat([
            bby_data, std_series, upper_band_series, lower_band_series,
            lookback_series, no_lb_period_series
        ], axis=1)
        bby_data.columns = [
            'sma', 'signal', 'std', 'upper_band', 'lower_band',
            'lookback', 'no_lb_period'
        ]
        return bby_data
    
    def get_individual_stream(self, spots : pd.Series):
        """
        Gets the Bollinger Bands signals from a given feed for a single trading day.

        Parameters
        ----------
        feed : pandas.DataFrame
            Financial feed processed with `feed.feed_processor`, filtered to a single trading day.
        
        Returns
        -------
        sma : numpy.ndarray
            Simple Moving Average values.
        std : numpy.ndarray
            Standard deviation values.
        rolling_counts : numpy.ndarray
            The number of ticks in the lookback window.
        no_lb_period : numpy.ndarray
            Ticks without a defined default or secondary lookback window.

        Notes
        -----
        * Assumes no null values in feed;
        * Assume first point is market open;
        * Uses first tick each rebalancing period for each period's signal.
        """
        # spots_array = spots.to_numpy()
        
        # Find lookback periods
        nearest_trading_start = spots.index[0]
        
        # Isolate when lookback period isn't constructed and when we use the secondary lookback period
        no_lb_period = (spots.index - nearest_trading_start) < pd.Timedelta(f'{self.lookback}s')

        # Consider rolling counts 
        rolling_counts = spots.rolling(pd.Timedelta(seconds=self.lookback)).count().to_numpy()

        # Get moving averages and stds:
        sma = spots.rolling(pd.Timedelta(seconds=self.lookback)).mean().to_numpy()
        std = spots.rolling(pd.Timedelta(seconds=self.lookback)).std().to_numpy()
        # Find index of first valid tick
        # first_publishable_tick = np.argmax(no_lb_period) + 1

        # Calculate SMA and standard deviation
        # sma, std = SMA_with_std(spots_array, rolling_counts, first_publishable_tick)
        
        return sma, std, rolling_counts, no_lb_period

_SIGNALS : dict[str, type[Signal]] = {
    'discrete' : Discrete_Var
}
