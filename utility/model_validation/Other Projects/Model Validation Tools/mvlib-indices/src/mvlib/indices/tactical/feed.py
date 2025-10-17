import numpy as np
import pandas as pd
import pytz
import datetime as dt
from .utils import file_filter

def process_bbg_csv(path : str, name : str, unit : str = 's', emitted : bool = True) -> pd.DataFrame:
    '''
    Processes Bloomberg sourced feed csvs
    Only filters columns and converts epochs to dates

    Parameters
    ----------
    path : str
        Path of the target csv
    name : str
        Name of the index
    unit : str, optional
        Unit of time of the epoch column. This value is represented by an offset alias, defaulting to seconds `s`.
    emitted : bool, optional
        If True, the data is assumed to have an emitted flag and only allow the emitted ticks. Ignores the emitted flag if set to False. 
        Older feeds (2024 First half and older) require `emitted` to be False. The default is True

    Returns
    -------
    bbg_csv : pandas.DataFrame.
        The processed Bloomberg CSV, with non-emitted ticks filtered out if applicable.
    '''
    df = pd.read_csv(path, usecols=[0,1,2,4] + ([7] if emitted else []), header=None)
    df.columns = ['epoch', 'bid', 'ask', 'spot'] + (['emitted'] if emitted else [])
    df.index = pd.to_datetime(df.epoch, unit=unit)
    df.index.name = name
    df.drop(columns=['epoch'], inplace=True)
    
    # Filters out ticks that are not used by the tactical
    # Determined by Deriv's feed filtering system
    if emitted:
        df['emitted'] = df['emitted'].apply(lambda x : int(x.replace(' ', '').split('|')[2][-1]))
        # print(df[~df['emitted']])
        df = df[df['emitted'] != 0]
    return df
    

def read_bbg_csvs(paths : str | list[str] | tuple[str], name, unit = 's', emitted=True):
    '''
    Reads and processes multiple Bloomberg sourced feed csvs.

    Parameters
    ----------

    paths : str | list[str] | tuple[str]
        Path of the target directory, csv or list of either
    name : str
        Name of the index
    unit : str, optional
        Unit of time of the epoch column. This value is represented by an offset alias, defaulting to seconds `s`.
    emitted : bool, optional
        If True, the data is assumed to have an emitted flag and only allow the emitted ticks. Ignores the emitted flag if set to False. 
        Older feeds (2024 First half and older) require `emitted` to be False. The default is True

    Returns
    -------
    bbg_csv : pandas.DataFrame.
        The processed Bloomberg CSV, with non-emitted ticks filtered out if applicable. 
        The csv is ordered with respect to the timestamps, regardless of order if `paths` is a list.
    '''
    # Multiple file handling
    file_paths = file_filter(paths)

    df_generator = (process_bbg_csv(filepath, name, unit, emitted) for filepath in file_paths)
    df = pd.concat(df_generator).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df

def is_dst(ts : pd.DatetimeIndex, tz : dt.tzinfo = pytz.UTC)-> pd.Series:
    """ 
    Checks if times provided are in US Daylight Savings Time or Standard Time. 
    This function takes MUCH LONGER than polars to run.

    Parameters
    ----------
    ts : pandas.DateTimeIndex
        A DateTimeIndex, possibly already localized, to check its DST status.
    tz : datetime.tzinfo
        Base timezone information of the index, if not already embedded within the index. This defaults to UTC.

    Returns
    -------
    dst_flags : pandas.Series
        A series of integers representing DST time if 1 or Standard Time if 0
    """
    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    ts_converted = ts.tz_convert("US/Eastern")
    dst_flag = ts_converted.map(lambda x : int(x.dst().total_seconds()!=0))
    ts = ts.tz_localize(None)
    return pd.Series(dst_flag, index = ts)

def specify_trading_times(feed : pd.DataFrame, tz=pytz.UTC, underlying_type='commodities'):
    """
    Specifies trading times of forex/commodity indices.
    
    Commodity (metal) indices.

    **Trading times (No DST)**:  
    > Mon-Thu -- 00:00:00 - 21:59:59, 23:00:00 - 23:59:59,  
    > Fri -- 00:00:00 - 21:54:59,  
    > Sat -- No Trading,  
    > Sun -- No Trading.
    
    **Trading times (DST)**:  
    > Mon-Thu -- 00:00:00 - 20:59:59. 22:00:00 - 23:59:59,  
    > Fri -- 00:00:00 - 20:54:59,  
    > Sat -- No Trading,  
    > Sun -- No Trading.

    Forex indices.

    **Trading times (No DST)**:  
    > Mon-Thu -- 00:00:00 - 21:59:59, 22:05:00 - 23:59:59,  
    > Fri -- 00:00:00 - 21:54:59,  
    > Sat -- No Trading,  
    > Sun -- 22:05:00 - 23:59:59.
    
    **Trading times (DST)**:  
    > Mon-Thu -- 00:00:00 - 20:59:59. 21:05:00 - 23:59:59,  
    > Fri -- 00:00:00 - 20:54:59,  
    > Sat -- No Trading,  
    > Sun -- 21:05:00 - 23:59:59.

    Forex indices (no breaks).

    **Trading times (No DST)**:  
    > Mon-Thu -- 00:00:00 - 23:59:59,  
    > Fri -- 00:00:00 - 21:54:59,  
    > Sat -- No Trading,  
    > Sun -- 22:05:00 - 23:59:59.
    
    **Trading times (DST)**:  
    > Mon-Thu -- 00:00:00 - 23:59:59,
    > Fri -- 00:00:00 - 20:54:59,  
    > Sat -- No Trading,  
    > Sun -- 21:05:00 - 23:59:59.

    Crypto indices.

    **Trading times (No DST)**:  
    > 24/7
    
    **Trading times (DST)**:  
    > 24/7

    Deriv-specific trading breaks are not considered.

    Parameters
    ----------
    feed : pandas.DataFrame
        The feed Dataframe, with a DateTimeIndex
    tz : datetime.tzinfo
        Base timezone information of the index, if not already embedded within the DateTimeIndex. This defaults to UTC.
    underlying_type : str
        Optional argument to configure market breaks. Must be one of [`commodities`, `forex`, `forex_no_breaks`]. Defaults to `commodities`.

    Returns
    -------
    deriv_trading_breaks : pandas.DataFrame
        A pandas DataFrame with a `trading_times` column representing open market hours and `dst_shift` for Daylight-Savings information.
    """
    index : pd.DatetimeIndex = feed.index
    dow = index.dayofweek
    h = index.hour
    m = index.minute

    localized_index = index.tz_localize(tz).tz_convert("US/Eastern")
    dst_shift = localized_index.map(lambda x : int(x.dst().total_seconds() != 0)) # 1 if DST, 0 if not

    if underlying_type == 'commodities':
        break_time = (h == 22 - dst_shift)
        friday_close = (dow == 4) & (((h == 21 - dst_shift) & (m >= 55))
                                    | (h >= 22 - dst_shift))
        saturday_close = (dow == 5)
        sunday_close = (dow == 6)
        weekend_close = friday_close | saturday_close | sunday_close
    elif underlying_type == 'forex':
        break_time = (h == 22 - dst_shift) & (m < 5)
        friday_close = (dow == 4) & (((h == 21 - dst_shift) & (m >= 55))
                                    | (h >= 22 - dst_shift))
        saturday_close = (dow == 5)
        sunday_close = (dow == 6) & ((h <= 21 - dst_shift) 
                                    | ((h == 22 - dst_shift) & (m < 5)))
        weekend_close = friday_close | saturday_close | sunday_close
    elif underlying_type == 'forex_no_breaks':
        break_time = (h != h)
        friday_close = (dow == 4) & (((h == 21 - dst_shift) & (m >= 55))
                                    | (h >= 22 - dst_shift))
        saturday_close = (dow == 5)
        sunday_close = (dow == 6) & ((h <= 21 - dst_shift) 
                                    | ((h == 22 - dst_shift) & (m < 5)))
        weekend_close = friday_close | saturday_close | sunday_close
    elif underlying_type == 'crypto':
        break_time = (h != h)
        weekend_close = (h != h)
    else:
        raise ValueError(f'Invalid underlying type. Must be one of [commodities, forex, forex_no_breaks, crypto]')
    
    market_break = break_time | weekend_close

    return pd.DataFrame({'trading_times' : ~market_break,
                         'dst_shift' : dst_shift},
                        index=index)

def feed_processor(feed : pd.DataFrame, underlying_type = 'commodities', lookback : int = None) -> pd.DataFrame:
    '''
    Returns a processed feed with trading times/days and missing tick indicators. 
    Missing bid/ask quotes are assumed to be the previously obtained bid/ask quotes. 
    Each trading day is represented as a distinct positive number.

    Parameters
    ----------
    feed : pandas.DataFrame
        The feed Dataframe, with a DateTimeIndex
    trading_times_bool : pandas.Series
        Trading times generated from :ref:`specify_trading_times`
    underlying_type : str
        Optional argument to configure market breaks. Must be one of [`commodities`, `forex`, `forex_no_breaks`]. Defaults to `commodities`.
    lookback : str
        Optional argument to set minimum length of break required for index to reset in seconds. Does nothing if set to None (Default).

    TODO: Set a test up so that it doesn't have any breaks except for weekend, check volatility on those 5 minutes.
    
    Returns
    -------
    processed_feed : pandas.DataFrame
        A pandas DataFrame containing the feed and trading day (`trading_day` column) information. 
        * Each trading day is a distinct positive integer.
        * Quotes in market close periods are dropped from the feed.
    '''
    data = feed.copy()
    trading_times_bool = specify_trading_times(feed, underlying_type=underlying_type)

    # Required to ensure the index knows when the end of the day is
    dst_shift = trading_times_bool['dst_shift']

    # Determine market reset periods
    if underlying_type in ['commodities', 'forex']:
        offsets = dst_shift.apply(lambda x : dt.time(hour=22 - x, minute=0)).astype(str)
        market_reset_placeholders_index = pd.to_datetime(np.char.add(data.index.date.astype(str), " ") + offsets).unique()
        market_reset_placeholders = pd.Series(0, index=market_reset_placeholders_index)
    elif underlying_type in ['forex_no_breaks', 'crypto']:
        market_reset_placeholders = None
    else:
        raise ValueError(f'Invalid underlying type. Must be one of [commodities, forex, forex_no_breaks, crypto]')
    
    # Determine long periods without ticks
    if lookback is not None:
        diffs = feed.index.to_series().diff()
        long_break_index = feed.index[(diffs > f'{lookback}s')]
        diff_reset_placeholders = pd.Series(0, index=long_break_index - pd.Timedelta('0.1ms'))
    else:
        diff_reset_placeholders = None

    # Get times when trading is open
    trading_day_opens = pd.concat([trading_times_bool['trading_times'], market_reset_placeholders, diff_reset_placeholders]).sort_index().astype(int).diff() > 0
    trading_day_opens = trading_day_opens[~trading_day_opens.index.duplicated(keep='first')]
    # Specify trading days (positive integers) and non-trading times (zero)
    data['trading_day'] = (trading_day_opens.cumsum().fillna(0)+1).where(trading_times_bool['trading_times'], 0)

    # Ensure no non-trading hours points
    data.loc[(data['trading_day'] == 0), data.columns.drop('trading_day')] = np.nan
    data.dropna(inplace=True)
    return data

"""
LT's (Structuring) Original Implementation
"""

def __get_trading_breaks(df: pd.DataFrame):
    dow = df.index.dayofweek
    h = df.index.hour
    m = df.index.minute
    dst_flag = is_dst(df.index).to_numpy().astype(bool) 

    # Market trading break here changes with asset
                           # During DST, closes on every Friday 20:55
    market_trading_break = ((dst_flag) * ((dow == 4) * (h == 20) * (m >= 55) +
                                         # Closes until Sunday night 22:05
                                          (dow == 4) * (h >= 21) +
                                          (dow == 5) +
                                          (dow == 6) * (h <= 21) +
                                          (dow == 6) * (h == 22) * (m < 5) +
                                         # During DST, closes every night 21:00 and reopens at 22:00
                                          (h == 21)) +
                           # When there is no DST, closes on every Friday 21:55
                           (~dst_flag) * ((dow == 4) * (h == 21) * (m >= 55) +
                                         # Closes until Sunday night 23:05
                                          (dow == 4) * (h >= 22) +
                                          (dow == 5) +
                                          (dow == 6) * (h <= 22) +
                                          (dow == 6) * (h == 23) * (m < 5) +
                                          # When there is no DST, closes every night 22:00 and reopens at 23:00
                                          (h == 22)))
                           # No need to include US holidays because Bloomberg will still populate data from other countries
    deriv_trading_break = (market_trading_break + 
                             (dow == 6) +
                             (dst_flag) * (dow == 4) * (h == 20) * (m >= 45) + 
                             (~dst_flag) * (dow == 4) * (h == 21) * (m >= 45))
    return market_trading_break, deriv_trading_break

def __dynamic_lb(df: pd.DataFrame,
               default_lb: int,
               opening_lb: int,
              ) -> tuple[pd.Series, np.array, pd.Series]:
    isna = df.squeeze().isna()
    na_cum_sum = isna.groupby((~isna).cumsum()).cumsum()
    reset = (na_cum_sum == 0) * (na_cum_sum.shift(1) >= default_lb)
    counter = reset.groupby(reset.cumsum()).cumcount() + 1 

    no_lb_mask = (counter < opening_lb) # reuse last_close 
    opening_lb_mask = counter < default_lb

    lookback_period = counter.copy()
    lookback_period.loc[opening_lb_mask] = opening_lb
    lookback_period = np.clip(lookback_period.to_numpy(), a_min = None, a_max = default_lb)
    
    return counter, lookback_period, no_lb_mask