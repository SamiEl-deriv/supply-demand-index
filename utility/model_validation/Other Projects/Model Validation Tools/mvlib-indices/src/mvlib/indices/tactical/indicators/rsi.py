import numpy as np
import numba as nb
import pandas as pd
from .base import Signal, Strategy
from .signals import Discrete_Const, fill_zeros_with_last
from .ma import SMMA

class RSI(Strategy):
    """
    Tactical RSI Strategy Implementation.
    
    Attributes
    ----------
    lookback : int
        Lookback period (s).
    secondary_lookback : int
        Secondary lookback period (s).
    upper_threshold : float
        Upper threshold (overbought). Typically above 50.
    lower_threshold : float
        Lower threshold (oversold). Typically below 50.
    rebalancing : int
        Period of time (s) between each rebalancing, i.e observation of RSI value for signal calculation. Defaults to 1.
    hold : bool
        Behaviour when RSI is between thresholds. Defaults to True.
        * Enabling `hold_position` is equivalent to BO's setting of hold (hold previous position);
        * Disabling this is equivalent to BO's setting of long cash (zero out position).
    signal_func : Signal
        The signal function, determining the position to take at each tick.

    Notes
    -----
    * Uses first tick each second for that second's signal;
    * Uses fixed thresholds.
    """

    num_of_underlyings = 1

    def __init__(self, lookback : int, secondary_lookback : int, thresholds : list[float], 
                 hold_position = True, rebalancing : int = 1, signal_func : str = 'discrete') -> None:
        """
        Parameters
        ----------
        lookback : int
            Lookback period (s).
        secondary_lookback : int
            Secondary lookback period (s).
        thresholds : list[float]
            List of thresholds for RSI signal: [upper_threshold, lower_threshold].
        hold_position : bool
            Behaviour when RSI is between thresholds. Defaults to True.
            * Enabling `hold_position` is equivalent to BO's setting of hold (hold previous position);
            * Disabling this is equivalent to BO's setting of long cash (zero out position).
        rebalancing : int
            Period of time (s) between each rebalancing, i.e observation of RSI value for signal calculation. Defaults to 1.
        """
        self.lookback = lookback
        self.secondary_lookback = secondary_lookback
        self.upper_threshold = thresholds['upper_threshold']
        self.lower_threshold = thresholds['lower_threshold']
        self.rebalancing = rebalancing
        self.hold = hold_position
        # Initialize signal function
        self.signal_func = _SIGNALS[signal_func](**thresholds)

    def get_signal(self, feed : pd.DataFrame) -> pd.DataFrame:
        """
        Gets the RSI signals from a given feed.

        Parameters
        ----------
        feed : pandas.DataFrame
            Financial feed processed with `feed.feed_processor`.
        
        Returns
        -------
        df_signal : pandas.DataFrame
            Pandas DataFrame containing the following columns:
            * `rsi` - RSI value;
            * `signal` - Output of signal function;
            * `lookback` - The number of ticks in the lookback window;
            * `up_smma` - SMMA of positive returns;
            * `down_smma - SMMA of negative returns;
            * `rolling_counts_default_lb` - The number of ticks in the default window;
            * `rolling_counts_opening_lb` - The number of ticks in the secondary window;
            * `no_lb_period` - Ticks without a defined default or secondary lookback window;
            * `default_lb_period` - Ticks without a defined default window, but with a defined secondary window.

        Notes
        -----
        * Assumes no null values in feed;
        * Uses first tick each second for that second's signal.
        """
        # Rebalanced index behaviour may break for durations not a factor of 60.
        rebalancing = f'{self.rebalancing}s'
        rebalanced_index = feed.index.floor(rebalancing).drop_duplicates(keep='first')

        # Define all required series
        rsi_series = pd.Series(np.nan, index=feed.index, name='rsi') 
        smma_up_series = pd.Series(np.nan, index=feed.index, name='smma_up')
        smma_down_series = pd.Series(np.nan, index=feed.index, name='smma_down')
        rolling_counts_default_lb_series = pd.Series(np.nan, index=feed.index, name='rolling_counts_default_lb')
        rolling_counts_secondary_lb_series = pd.Series(np.nan, index=feed.index, name='rolling_counts_secondary_lb')
        no_lb_period_series = pd.Series(None, index=feed.index, name='no_lb_period', dtype=bool)
        secondary_lb_period_series = pd.Series(None, index=feed.index, name='secondary_lb_period', dtype=bool)
        lookback_series = pd.Series(np.nan, index=feed.index, name='lookback')
        signal_series = pd.Series(np.nan, index=rebalanced_index, name='signal').asfreq(rebalancing)
        
        # Filter to process on each trading day
        for name, group in feed.groupby(['trading_day']):
            # Remove non-trading times
            if name[0] == 0:
                continue

            # Retrieve trading times per day
            times = group.index
            # Get RSI & additional data per stream
            rsi_piece_raw, lookback, smma_up, smma_down, rolling_counts_default_lb, rolling_counts_secondary_lb, no_lb_period, secondary_lb_period = self.get_individual_stream(group['spot'])

            # Assign obtained data to proper trading times 
            rsi_series[times] = rsi_piece_raw
            no_lb_period_series[times] = no_lb_period
            secondary_lb_period_series[times] = secondary_lb_period
            lookback_series[times] = lookback
            smma_up_series[times] = smma_up
            smma_down_series[times] = smma_down
            rolling_counts_default_lb_series[times] = rolling_counts_default_lb
            rolling_counts_secondary_lb_series[times] = rolling_counts_secondary_lb

            # Cast RSI to the rebalancing period time-frame
            rsi_at_rebalancing = pd.Series(rsi_piece_raw, index=times)[~no_lb_period]
            rsi_at_rebalancing.index = rsi_at_rebalancing.index.floor(rebalancing)
            rsi_at_rebalancing = rsi_at_rebalancing[~rsi_at_rebalancing.index.duplicated(keep='first')]

            # Get effective portfolio
            signal_piece = self.signal_func(rsi_at_rebalancing)
            # Where the position would normally be closed, it preserves the position instead
            if self.hold:
                signal_piece = fill_zeros_with_last(signal_piece)
            signal_series[rsi_at_rebalancing.index] = signal_piece
        # Combine data
        rsi_data = pd.merge_asof(rsi_series, signal_series, left_index=True, right_index=True, direction='backward')
        rsi_data = pd.concat([rsi_data, lookback_series, smma_up_series, smma_down_series, 
                              rolling_counts_default_lb_series, rolling_counts_secondary_lb_series,
                              no_lb_period_series, secondary_lb_period_series], axis=1)
        rsi_data.columns = ['rsi', 'signal', 'lookback', 'up_smma', 'down_smma',
                            'rolling_counts_default_lb', 'rolling_counts_secondary_lb',
                            'no_lb_period', 'secondary_lb_period']
        return rsi_data
    
    def get_individual_stream(self, spots : pd.Series):
        """
        Gets the RSI signals from a given feed for a single trading day.

        Parameters
        ----------
        feed : pandas.DataFrame
            Financial feed processed with `feed.feed_processor`, filtered to a single trading day.
        
        Returns
        -------
        rsi : numpy.ndarray
            RSI values.
        rolling_counts : numpy.ndarray
            The number of ticks in the lookback window.
        smma_up : numpy.ndarray
            SMMA of positive returns.
        smma_down : numpy.ndarray
            SMMA of negative returns.
        rolling_counts_default_lb pandas.Series
            The number of ticks in the default window.
        rolling_counts_secondary_lb pandas.Series
            The number of ticks in the secondary window.
        no_lb_period : numpy.ndarray
            Ticks without a defined default or secondary lookback window.
        secondary_lb_period : numpy.ndarray
            Ticks without a defined default window, but with a defined secondary window.

        Notes
        -----
        * Assumes no null values in feed;
        * Assume first point is market open;
        * Uses first tick each rebalancing period for each period's signal.
        """
        lookback = self.lookback
        secondary_lookback = self.secondary_lookback
        diffs = spots.diff().fillna(0).to_numpy()
        neutral_rsi = (self.upper_threshold + self.lower_threshold) / 2
        zero_down_level = 100
        rsi = np.full_like(diffs, neutral_rsi)

        ups = np.maximum(diffs, 0)
        downs = np.maximum(-diffs, 0)

        # Find lookback periods
        # This is what BE is currently implementing, where the first obtained point is considered the start of the lookback period
        nearest_trading_start = spots.index[0]
        
        # Isolate when lookback period isn't constructed and when we use the secondary lookback period
        no_lb_period = (spots.index - nearest_trading_start) < pd.Timedelta(f'{secondary_lookback}s')
        secondary_lb_period =  ~no_lb_period & ((spots.index - nearest_trading_start) < pd.Timedelta(f'{lookback}s'))

        # Consider rolling counts 
        rolling_counts_default_lb = spots.rolling(pd.Timedelta(seconds=lookback)).count()
        rolling_counts_secondary_lb = spots.rolling(pd.Timedelta(seconds=secondary_lookback)).count()
        rolling_counts = rolling_counts_secondary_lb.where(secondary_lb_period, rolling_counts_default_lb).to_numpy()

        # Find index of first valid tick
        first_publishable_tick = np.argmax(no_lb_period) + 1
        # If index of first valid tick is out of bounds, return RSI as is
        if first_publishable_tick > len(no_lb_period):
            return rsi
        
        # Get SMMAs base on the weights from the rolling counts
        smma_up = SMMA(ups, rolling_counts, first_publishable_tick)
        smma_down = SMMA(downs, rolling_counts, first_publishable_tick)
            
        # Calculate RSI & generate NaNs if ticks are invalid
        down_valid_mask = smma_down != 0
        up_valid_mask = smma_up != 0
        rs = np.empty_like(smma_down)
        rs[down_valid_mask] = smma_up[down_valid_mask] / smma_down[down_valid_mask]
        rs[~down_valid_mask & up_valid_mask] = zero_down_level
        rs[~(down_valid_mask | up_valid_mask)] = np.nan
        rsi = 100 - 100 / (1 + rs)
        return rsi, rolling_counts, smma_up, smma_down, rolling_counts_default_lb, rolling_counts_secondary_lb, no_lb_period, secondary_lb_period

_SIGNALS : dict[str, type[Signal]] = {
    'discrete' : Discrete_Const
}
