import numpy as np
import pandas as pd
from .feed import feed_processor
from .indicators.base import Strategy
from .indicators.rsi import RSI
from .indicators.bby import BBY
from .indicators.maco import MACO
from .indicators.pt import PT

STRATEGIES : dict[str, Strategy] = {
    'RSI' : RSI,
    'BB' : BBY,
    'MACO' : MACO,
    'PT' : PT
}

class Tactical:
    """
    Base Tactical Index Generator. Supports strategies indexed in `STRATEGIES`.

    Attributes
    ----------
    strategy : Strategy
        Underlying strategy for Tactical Index.
    lookback : int
        Lookback period (s).
    rebalancing : int
        Period of time (s) between each rebalancing, i.e observation of signal value for calculation.
    spread : dict
        Specs for spread. Requires floats:
        * 0 < alpha;
        * calibration set by quants;
        * commission set by dealer (>0).
    initial : float
        Initial value of tactical index.
    name : float
        Name of tactical index.
    """
    def __init__(self, strategy_type: str, spread: dict, initial: float, name: str, subtype: str, leverages: dict, **strategy_params) -> None:
        """
        Parameters
        ----------
        strategy_type : str
            Strategy of Tactical Index. Must be in `STRATEGIES`.
        spread : dict
            Specs for spread. Requires floats:
            * 0 < alpha;
            * calibration set by quants;
            * commission set by dealer (>0).
        initial : float
            Initial value of tactical index.
        name : str
            Name of tactical index.
        **strategy_params
            Strategy parameters required for strategy given by `strategy_type`.
            Refer to each strategy for required values.
        """
        self.strategy = STRATEGIES[strategy_type](**strategy_params)
        self.lookback = strategy_params['lookback']
        self.rebalancing = strategy_params['rebalancing']
        self.subtype = subtype
        self.upper_leverage = leverages['upper_leverage']
        self.lower_leverage = leverages['lower_leverage']
        self.spread = spread
        self.initial = initial
        self.name = name

    def _get_underlying_type(self, underlying_name: str) -> str:
        """
        Determines the underlying type based on the underlying name.
        
        Parameters
        ----------
        underlying_name : str
            Name of the underlying.
            
        Returns
        -------
        str
            The underlying type ('crypto', 'commodities', or 'forex_no_breaks').
        """
        if 'cry' in underlying_name:
            return 'crypto'
        elif 'frx' in underlying_name:
            if any(s in underlying_name for s in ['XAU', 'XAG', 'XPT']):
                return 'commodities'
            else:
                return 'forex_no_breaks'
        else:
            raise ValueError('Invalid underlying. Underlying must contain one of [cry, frx]')

    def _process_feed(self, raw_feed: pd.DataFrame, underlying_type: str, start: str, end: str, blank_periods: bool) -> pd.DataFrame:
        """
        Process a raw feed into the required format.
        
        Parameters
        ----------
        raw_feed : pandas.DataFrame
            Raw feed data.
        underlying_type : str
            Type of the underlying.
        start : str
            Start date.
        end : str
            End date.
        blank_periods : bool
            Whether to include blank periods.
            
        Returns
        -------
        pandas.DataFrame
            Processed feed.
        """
        feed = feed_processor(raw_feed.loc[start:end,:], underlying_type=underlying_type, 
                            lookback=self.lookback if blank_periods else None)
        feed = feed.drop(columns=['bid', 'ask'])
        return feed

    def _calculate_tactical_index_single(self, feed: pd.DataFrame) -> np.ndarray:
        """
        Calculate the tactical index values.
        
        Parameters
        ----------
        feed : pandas.DataFrame
            Processed feed data.
        spots : pandas.Series
            Spot prices.
            
        Returns
        -------
        numpy.ndarray
            Tactical index values.
        """
        tactical_index = np.full_like(feed['spot'], self.initial)
        returns = self._get_spot_returns(feed, 'spot', return_type='simple')
        leverage_long = np.maximum(feed['leverage'], 0)
        leverage_short = np.maximum(-feed['leverage'], 0)
        return tactical_index * np.nancumprod((1 + leverage_long * returns) / (1 + leverage_short * returns))
        
    def _calculate_tactical_index_pair(self, feed: pd.DataFrame) -> np.ndarray:
        """
        Calculate the tactical index values.
        
        Parameters
        ----------
        feed : pandas.DataFrame
            Processed feed data.
        spots : pandas.Series
            Spot prices.
            
        Returns
        -------
        numpy.ndarray
            Tactical index values.
        """
        tactical_index = np.full_like(feed['spot_1'], self.initial)
        spots_1 = feed['spot_1']
        spots_2 = feed['spot_2']
        returns_1 = self._get_spot_returns(feed, 'spot_1', return_type='log')
        returns_2 = self._get_spot_returns(feed, 'spot_2', return_type='log')
        leverage_1 = feed['leverage_1']
        leverage_2 = feed['leverage_2']
        return tactical_index * np.exp(np.nancumsum((leverage_1 * spots_1 * returns_1 + leverage_2 * spots_2 * returns_2) /
                                                    (leverage_1.abs() * spots_1 + leverage_2.abs() * spots_2)))

    def _get_spot_returns(self, feed: pd.DataFrame, spot_col: str, return_type : str = 'simple') -> np.ndarray:
        """
        Get simple returns while considering non-trading times.
        
        Parameters
        ----------
        feed : pandas.DataFrame
            Processed feed data.
        spot_col : str
            Name of the spot column.
        return_type : str
            Type of return, must be one of [simple, log]
            
        Returns
        -------
        numpy.ndarray
            Returns array.
        """
        returns_series = pd.Series(index=feed.index, name='Returns')
        filled_spots = feed[spot_col].ffill()
        for name, group in feed.groupby('trading_day'):
            if name == 0:
                continue
            times = group.index
            filled_spots_piece = filled_spots[times]
            if return_type == 'simple':
                returns = np.diff(filled_spots_piece) / filled_spots_piece[:-1]
            elif return_type == 'log':
                returns = np.log(filled_spots_piece[1:]) - np.log(filled_spots_piece[:-1])
            else:
                raise ValueError(f'Expected return type to be one of ["simple", "log"] but got {return_type}')
            returns_series[times[1:]] = returns
        return returns_series.to_numpy()
    

    def _apply_spread(self, tactical_values: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        Apply spread to tactical values to get bid/ask prices.
        
        Parameters
        ----------
        tactical_values : pandas.Series
            Tactical index values.
            
        Returns
        -------
        tuple[pandas.Series, pandas.Series]
            Bid and ask prices.
        """
        spread = self.spread['alpha'] * self.spread['calibration'] + self.spread['commission']
        bid = tactical_values * (1 - spread / 2)
        ask = tactical_values * (1 + spread / 2)
        return bid, ask


class Tactical_Single(Tactical):
    """Single asset tactical index generator."""
    
    def __init__(self, strategy_type, spread, initial, name, subtype, leverages, **strategy_params):
        # Check to see that strategy is appropriate
        if n := STRATEGIES[strategy_type].num_of_underlyings != 1:
            raise ValueError(f'Expected single-asset strategy. Got {strategy_type}, a {n}-asset strategy')
        super().__init__(strategy_type, spread, initial, name, subtype, leverages, **strategy_params)

    def generate_tactical(self, raw_feed: pd.DataFrame, underlying_name: str, start: str, end: str, 
                        blank_periods=True, details=False) -> pd.DataFrame:
        """
        Generate tactical index for a single asset.
        
        Parameters
        ----------
        raw_feed : pandas.DataFrame
            Bloomberg feed with a DateTimeIndex containing the columns [spot, bid, ask].
        underlying_name : str
            Name of the underlying.
        start : str
            The start date in `YYYY-MM-DD hh:mm:ss` format.
        end : str
            The end date in `YYYY-MM-DD hh:mm:ss` format.
        blank_periods : bool, optional
            Whether to include blank periods, by default True.
        details : bool, optional
            Returns additional columns, by default False.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing tactical index output.
        """
        underlying_type = self._get_underlying_type(underlying_name)
        feed = self._process_feed(raw_feed, underlying_type, start, end, blank_periods)
        
        # Get signals
        tactical_data = self.strategy.get_signal(feed)
        feed = pd.concat([feed, tactical_data], axis=1)

        # Properly map leverages
        if self.subtype == 'trend':
            upper_leverage = self.upper_leverage
            lower_leverage = -self.lower_leverage
        elif self.subtype == 'contrarian':
            upper_leverage = -self.upper_leverage
            lower_leverage = self.lower_leverage
        else:
            raise ValueError(f'Invalid Subtype. Got {self.subtype}. Must be one of ["contrarian", "trend"].')
        
        feed['signal'] = feed['signal'].where(~feed['no_lb_period'], 0)
        feed['leverage'] = np.where(feed['signal'] == 1, upper_leverage, 
                                    np.where(feed['signal'] == -1, lower_leverage, 
                                             0))
        # Calculate tactical index
        tactical_index = self._calculate_tactical_index_single(feed)
        feed[self.name] = tactical_index
        
        # Apply spread
        feed['bid'], feed['ask'] = self._apply_spread(feed[self.name])
        
        # Final formatting
        feed.index.name = 'ts'
        feed.rename(columns={'spot': underlying_name}, inplace=True)
        
        return feed if details else feed[[underlying_name, self.name, 'bid', 'ask', 'no_lb_period']]


class Tactical_Pair(Tactical):
    """Pair of assets tactical index generator."""
    
    def __init__(self, strategy_type, spread, initial, name, subtype, leverages, **strategy_params):
        # Check to see that strategy is appropriate
        if n := STRATEGIES[strategy_type].num_of_underlyings != 2:
            raise ValueError(f'Expected double-asset strategy. Got {strategy_type}, a {n}-asset strategy')
        super().__init__(strategy_type, spread, initial, name, subtype, leverages, **strategy_params)

    def generate_tactical(self, raw_feeds: dict[str, pd.DataFrame], start: str, end: str,
                        blank_periods=True, details=False) -> pd.DataFrame:
        """
        Generate tactical index for multiple assets.
        
        Parameters
        ----------
        raw_feeds : dict[str, pandas.DataFrame]
            Dictionary mapping underlying names to their Bloomberg feeds.
            Each feed should have a DateTimeIndex containing the columns [spot, bid, ask].
        start : str
            The start date in `YYYY-MM-DD hh:mm:ss` format.
        end : str
            The end date in `YYYY-MM-DD hh:mm:ss` format.
        blank_periods : bool, optional
            Whether to include blank periods, by default True.
        details : bool, optional
            Returns additional columns, by default False.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing tactical index output.
        """

        underlying_names = list(raw_feeds.keys())

        if len(raw_feeds) != 2:
            raise ValueError(f"Expected a dict of two feeds, got {len(raw_feeds)} items instead.")
        if not all(isinstance(df, pd.DataFrame) for df in raw_feeds.values()):
            raise ValueError(f"Expected DataFrames, got {[type(df) for df in raw_feeds.values()]} instead.")
        
        if len(set(self._get_underlying_type(underlying_name) for underlying_name in underlying_names)) == 1:
            underlying_type = self._get_underlying_type(underlying_names[0])
        else:
            raise ValueError(f'Currently supports pairs of the same underlying type only. Got differing types of underlyings: {underlying_names}')
            
        # Combine feeds with outer merge
        feed = pd.merge(left=raw_feeds[underlying_names[0]]['spot'], right=raw_feeds[underlying_names[1]]['spot'], 
                        how='outer', left_index=True, right_index=True, suffixes=['_1','_2']).ffill().dropna()
        # Process feed
        feed = self._process_feed(feed, underlying_type, start, end, blank_periods)
        
        # Get signals using combined feed
        tactical_data = self.strategy.get_signal(feed)
        feed = pd.concat([feed, tactical_data], axis=1)

        # Properly map leverages
        if self.subtype == 'trend':
            upper_leverage = self.upper_leverage
            lower_leverage = -self.lower_leverage
        elif self.subtype == 'contrarian':
            upper_leverage = -self.upper_leverage
            lower_leverage = self.lower_leverage
        else:
            raise ValueError(f'Invalid Subtype. Got {self.subtype}. Must be one of ["contrarian", "trend"].')
        
        feed['signal'] = feed['signal'].where(~feed['no_lb_period'], 0)
        feed['leverage_1'] = np.where(feed['signal'] == 1, upper_leverage, 
                                    np.where(feed['signal'] == -1, -upper_leverage, 
                                             0))
        feed['leverage_2'] = np.where(feed['signal'] == 1, lower_leverage, 
                                    np.where(feed['signal'] == -1, -lower_leverage, 
                                             0))
        tactical_index = self._calculate_tactical_index_pair(feed)
        feed[self.name] = tactical_index
        
        # Apply spread
        feed['bid'], feed['ask'] = self._apply_spread(feed[self.name])
        
        # Final formatting
        feed.index.name = 'ts'
        
        feed.rename(columns=dict(zip(['spot_1', 'spot_2'], underlying_names)), inplace=True)

        columns_to_return = underlying_names + [self.name, 'bid', 'ask', 'no_lb_period']
        return feed if details else feed[columns_to_return]
