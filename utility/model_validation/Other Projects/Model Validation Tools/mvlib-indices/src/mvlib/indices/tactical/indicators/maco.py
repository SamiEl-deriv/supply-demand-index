import numpy as np
import numba as nb
import pandas as pd
from .base import Signal, Strategy
from .signals import Discrete_Const, fill_zeros_with_last

class MACO(Strategy):
    """
    Tactical Moving Average Crossover Strategy Implementation.
    
    Attributes
    ----------
    fast_lookback : int
        Fast moving average lookback period (s).
    lookback : int
        Slow moving average lookback period (s).
    upper_leverage : float
        Leverage when fast MA crosses above slow MA.
    lower_leverage : float
        Leverage when fast MA crosses below slow MA.
    rebalancing : int
        Period of time (s) between each rebalancing, i.e observation of crossovers for signal calculation. Defaults to 1.
    hold : bool
        Behaviour when MAs are equal. Defaults to True.
        * Enabling `hold_position` is equivalent to holding the previous position;
        * Disabling this is equivalent to zeroing out the position.
    signal_func : Signal
        The signal function, determining the position to take at each tick.

    Notes
    -----
    * Uses first tick each second for that second's signal;
    * Generates signals based on crossovers between fast and slow moving averages.
    """

    num_of_underlyings = 1

    def __init__(self, fast_lookback: int, lookback: int, hold_position=True, rebalancing: int=1, signal_func: str='discrete') -> None:
        """
        Parameters
        ----------
        fast_lookback : int
            Fast moving average lookback period (s).
        lookback : int
            Slow moving average lookback period (s).
        hold_position : bool
            Behaviour when MAs are equal. Defaults to True.
            * Enabling `hold_position` is equivalent to holding the previous position;
            * Disabling this is equivalent to zeroing out the position.
        rebalancing : int
            Period of time (s) between each rebalancing, i.e observation of crossovers for signal calculation. Defaults to 1.
        """
        if fast_lookback >= lookback:
            raise ValueError("fast_lookback must be less than lookback")
            
        self.fast_lookback = fast_lookback
        self.lookback = lookback
        self.rebalancing = rebalancing
        self.hold = hold_position
        
        # Initialize signal function with dummy thresholds since we'll use the MAs themselves
        self.signal_func = _SIGNALS[signal_func](upper_threshold = 0, lower_threshold = 0)

    def get_signal(self, feed: pd.DataFrame) -> pd.DataFrame:
        """
        Gets the Moving Average Crossover signals from a given feed.

        Parameters
        ----------
        feed : pandas.DataFrame
            Financial feed processed with `feed.feed_processor`.
        
        Returns
        -------
        df_signal : pandas.DataFrame
            Pandas DataFrame containing the following columns:
            * `fast_ma` - Fast Moving Average value;
            * `slow_ma` - Slow Moving Average value;
            * `signal` - Output of signal function;
            * `lookback` - The number of ticks in the lookback window;
            * `no_lb_period` - Ticks without a defined lookback window;
        """
        # Rebalanced index behaviour may break for durations not a factor of 60.
        rebalancing = f'{self.rebalancing}s'
        rebalanced_index = feed.index.floor(rebalancing).drop_duplicates(keep='first')

        # Define all required series
        fast_ma_series = pd.Series(np.nan, index=feed.index, name='fast_ma')
        slow_ma_series = pd.Series(np.nan, index=feed.index, name='slow_ma')
        no_lb_period_series = pd.Series(None, index=feed.index, name='no_lb_period', dtype=bool)
        fast_lookback_series = pd.Series(np.nan, index=feed.index, name='fast_lookback')
        lookback_series = pd.Series(np.nan, index=feed.index, name='lookback')
        signal_series = pd.Series(np.nan, index=rebalanced_index, name='signal').asfreq(rebalancing)
        
        # Filter to process on each trading day
        for name, group in feed.groupby(['trading_day']):
            # Remove non-trading times
            if name[0] == 0:
                continue

            # Retrieve trading times per day
            times = group.index
            # Get MAs & additional data per stream
            fast_ma, slow_ma, fast_lookback, lookback, no_lb_period = self.get_individual_stream(group['spot'])

            # Assign obtained data to proper trading times 
            fast_ma_series[times] = fast_ma
            slow_ma_series[times] = slow_ma
            no_lb_period_series[times] = no_lb_period
            fast_lookback_series[times] = fast_lookback
            lookback_series[times] = lookback

            # Cast MAs to the rebalancing period time-frame
            mas_at_rebalancing = pd.DataFrame({
                'fast_ma': fast_ma_series[times][~no_lb_period],
                'slow_ma': slow_ma_series[times][~no_lb_period]
            })
            mas_at_rebalancing.index = mas_at_rebalancing.index.floor(rebalancing)
            mas_at_rebalancing = mas_at_rebalancing[~mas_at_rebalancing.index.duplicated(keep='first')]

            # Get portfolio signal based on MA crossovers
            signal_piece = self.signal_func(mas_at_rebalancing['slow_ma'] - mas_at_rebalancing['fast_ma']
                                          )
            # Where the position would normally be closed, it preserves the position instead
            if self.hold:
                signal_piece = fill_zeros_with_last(signal_piece)
            signal_series[mas_at_rebalancing.index] = signal_piece

        # Combine data
        maco_data = pd.merge_asof(fast_ma_series, signal_series, left_index=True, right_index=True, direction='backward')
        maco_data = pd.concat([
            maco_data, slow_ma_series,
            fast_lookback_series, lookback_series, 
            no_lb_period_series
        ], axis=1)
        maco_data.columns = [
            'fast_ma', 'signal', 'slow_ma',
            'fast_lookback', 'lookback', 
            'no_lb_period'
        ]
        return maco_data
    
    def get_individual_stream(self, spots: pd.Series):
        """
        Gets the Moving Average signals from a given feed for a single trading day.

        Parameters
        ----------
        feed : pandas.DataFrame
            Financial feed processed with `feed.feed_processor`, filtered to a single trading day.
        
        Returns
        -------
        fast_ma : numpy.ndarray
            Fast Moving Average values.
        slow_ma : numpy.ndarray
            Slow Moving Average values.
        rolling_counts : numpy.ndarray
            The number of ticks in the lookback window.
        no_lb_period : numpy.ndarray
            Ticks without a defined lookback window.

        Notes
        -----
        * Assumes no null values in feed;
        * Assume first point is market open;
        * Uses first tick each rebalancing period for each period's signal.
        """
        nearest_trading_start = spots.index[0]
        
        # Isolate when lookback period isn't constructed
        no_lb_period = (spots.index - nearest_trading_start) < pd.Timedelta(f'{self.lookback}s')

        # Consider rolling counts (use slower MA for lookback since it needs more data)
        fast_rolling_counts = spots.rolling(pd.Timedelta(seconds=self.fast_lookback)).count().to_numpy()
        slow_rolling_counts = spots.rolling(pd.Timedelta(seconds=self.lookback)).count().to_numpy()

        # Get moving averages
        fast_ma = spots.rolling(pd.Timedelta(seconds=self.fast_lookback)).mean().to_numpy()
        slow_ma = spots.rolling(pd.Timedelta(seconds=self.lookback)).mean().to_numpy()
        
        return fast_ma, slow_ma, fast_rolling_counts, slow_rolling_counts, no_lb_period

_SIGNALS: dict[str, type[Signal]] = {
    'discrete': Discrete_Const
}
