import pandas as pd
from pandas.tseries.offsets import DateOffset
from pandas.core.indexes.period import PeriodIndex
import numpy as np
import copy
from scipy.special import erf
import scipy.stats as ss
from datetime import timedelta, datetime
import os
import warnings
from typing import Optional, Tuple, List, Union, Callable
from tqdm import tqdm
import matplotlib.pyplot as plt
import functools
from collections import OrderedDict
import time


warnings.filterwarnings('ignore')


def identity(x: Union[np.array, float]) -> Union[np.array, float]:
    return x


def calculate_shift(x: Union[np.array,
                             float],
                    y: Union[np.array,
                             float],
                    shift_type: Callable,
                    b: float = 1.0) -> Union[np.array,
                                             float]:
    """
    shift
    """

    if shift_type == np.log:
        # log shift
        return np.log(y) - np.log(x)

    elif shift_type == np.arcsinh:
        # arcsinh shift
        return np.arcsinh(y / b) - np.arcsinh(x / b)

    elif shift_type == identity:
        # simple(absolute) shift
        return y - x

    else:
        raise NotImplementedError


def norm2t_transform(fwd_fx: np.array, lb_fx: np.array, dof: int) -> Tuple[
        np.array, float, float, float]:
    """
    the index_change = student_t_inverse(Normal CDF of the FX change)
    fwd_fx : new FX changes/returns
    lb_fx : array of (lookback) historical changes/returns
    dof : the degrees of freedom for the student-t distribution
    """

    lb_mean = np.nanmean(lb_fx)
    lb_std = np.nanstd(lb_fx)

    index_change = ss.t.ppf(
        np.clip(
            ss.norm.cdf(
                (fwd_fx - lb_mean) / lb_std),
            1e-6,
            1 - 1e-6),
        df=dof)

    unique_val, counts = np.unique(
        lb_fx, return_index=False, return_inverse=False, return_counts=True)

    g = ss.t.ppf(
        np.clip(
            ss.norm.cdf(
                (unique_val - lb_mean) / lb_std),
            1e-6,
            1 - 1e-6),
        df=dof)
    prob_g = counts / len(lb_fx)

    # mean, std of g
    beta = np.sum(g * prob_g)
    alpha = np.sqrt(np.sum(np.power(g - beta, 2) * prob_g))

    return index_change, lb_mean, alpha, beta


def t2norm_transform(fwd_fx: np.array, lb_fx: np.array,
                     dof: int) -> Tuple[np.array, float, float, float]:
    """
    the index_change = norm_inverse(Student-t CDF of the FX change)
    fwd_fx : new FX changes/returns
    lb_fx : array of (lookback) historical changes/returns
    dof : the degrees of freedom for the student-t distribution
    """

    lb_mean = np.nanmean(lb_fx)
    lb_std = np.nanstd(lb_fx)

    index_change = ss.norm.ppf(
        np.clip(
            ss.t.cdf(
                ((fwd_fx - lb_mean) / lb_std),
                df=dof),
            1e-6,
            1 - 1e-6))

    unique_val, counts = np.unique(
        lb_fx, return_index=False, return_inverse=False, return_counts=True)

    g = ss.norm.ppf(
        np.clip(
            ss.t.cdf(
                ((unique_val - lb_mean) / lb_std),
                df=dof),
            1e-6,
            1 - 1e-6))
    prob_g = counts / len(lb_fx)

    # mean, std of g
    beta = np.sum(g * prob_g)
    alpha = np.sqrt(np.sum(np.power(g - beta, 2) * prob_g))

    return index_change, lb_mean, alpha, beta


def mixing_t2norm_norm2t(fwd_fx: np.array,
                         lb_fx: np.array,
                         t2norm_dof: int,
                         norm2t_dof: int,
                         threshold: float) -> Tuple[np.array,
                                                    float,
                                                    float,
                                                    float]:
    """
    left_threshold and right_threshold are expressed in terms of units of lb_std
    """
    assert (threshold > 0)

    lb_mean = np.nanmean(lb_fx)
    lb_std = np.nanstd(lb_fx)

    left_threshold = -1 * threshold * lb_std
    right_threshold = threshold * lb_std

    index_change = np.zeros(len(fwd_fx))

    select_cond = np.logical_or(
        fwd_fx > right_threshold, fwd_fx < left_threshold)
    index_change[select_cond] = ss.norm.ppf(
        np.clip(
            ss.t.cdf(
                (fwd_fx[select_cond] - lb_mean) / lb_std,
                df=t2norm_dof),
            1e-6,
            1 - 1e-6))

    select_cond = np.logical_and(
        fwd_fx <= right_threshold, fwd_fx >= left_threshold)
    index_change[select_cond] = ss.t.ppf(
        np.clip(
            ss.norm.cdf(
                (fwd_fx[select_cond] - lb_mean) / lb_std),
            1e-6,
            1 - 1e-6),
        df=norm2t_dof)

    unique_val, counts = np.unique(
        lb_fx, return_index=False, return_inverse=False, return_counts=True)
    g = np.zeros(len(unique_val))

    select_cond = np.logical_or(
        unique_val > right_threshold, unique_val < left_threshold)
    g[select_cond] = ss.norm.ppf(
        np.clip(
            ss.t.cdf(
                (unique_val[select_cond] - lb_mean) / lb_std,
                df=t2norm_dof),
            1e-6,
            1 - 1e-6))
    select_cond = np.logical_and(
        unique_val <= right_threshold, unique_val >= left_threshold)
    g[select_cond] = ss.t.ppf(
        np.clip(
            ss.norm.cdf(
                (unique_val[select_cond] - lb_mean) / lb_std),
            1e-6,
            1 - 1e-6),
        df=norm2t_dof)

    prob_g = counts / len(lb_fx)

    beta = np.sum(g * prob_g)
    alpha = np.sqrt(np.sum(np.power(g - beta, 2) * prob_g))

    return index_change, lb_mean, alpha, beta


class IntradaySeparation:
    """
    class that deals with intraday time splits, event time splits -- suitable for building lookback distribution as well
    """
    # set True if reusing returns in the event times for building
    # distributions across time seperations
    reuse_eventdata_4timesplit = True

    def __init__(self, ts: pd.Series,
                 log_returns: np.array,
                 simple_changes: np.array,
                 time_splits: List[Tuple[float, float]],
                 event: bool = False,
                 event_list: list = []
                 ):

        self.time_splits = time_splits
        self.event = event
        self.event_list = event_list

        if (len(log_returns) > 0) and (len(simple_changes) > 0):
            assert (len(log_returns) == len(simple_changes))

            self.df = pd.DataFrame(
                data=np.vstack(
                    (log_returns, simple_changes)).T, columns=[
                    'log_return', 'simple_return'], index=ts)

        elif (len(log_returns) > 0):
            self.df = pd.DataFrame(data=log_returns, columns=[
                                   'log_return'], index=ts)

        else:
            self.df = pd.DataFrame(data=simple_changes, columns=[
                                   'simple_change'], index=ts)

        # use dict in this version; consider in the future changing to using
        # dataframe
        self.record = dict()

    def make_event_dist(self,
                        pre_event_timedelta: timedelta,
                        post_event_timedelta: timedelta,
                        event_types: List[str]
                        ):

        if not (self.event):
            self.df.loc[:, 'event_time'] = False
            return None

        assert (set(self.event_list['easy']) <= set(event_types))

        for event_type in event_types:
            self.record['event_' + event_type + '_log_return'] = np.array([])
            self.record['event_' + event_type +
                        '_simple_change'] = np.array([])

        # initialization of event time. False by default
        self.df.loc[:, 'event_time'] = False

        # store the loc/returns per event
        for i in range(len(self.event_list)):

            event_type, event_time = self.event_list.loc[i, [
                'easy', 'time']].values
            start_t = event_time + pre_event_timedelta
            end_t = event_time + post_event_timedelta
            loc = np.where(np.logical_and(self.df.index >=
                           start_t, self.df.index <= end_t))[0]

            self.record['event_' + str(i) + '_loc'] = loc

            if 'log_return' in self.df.columns:
                self.record['event_' +
                            str(i) +
                            '_log_return'] = self.df.log_return.values[loc]
                self.record['event_' +
                            event_type +
                            '_log_return'] = np.concatenate([self.record['event_' +
                                                                         str(i) +
                                                                         '_log_return'], self.record['event_' +
                                                            event_type +
                                                            '_log_return']])

            if 'simple_change' in self.df.columns:
                self.record['event_' +
                            str(i) +
                            '_simple_change'] = self.df.simple_change.values[loc]
                self.record['event_' +
                            event_type +
                            '_simple_change'] = np.concatenate([self.record['event_' +
                                                                            str(i) +
                                                                            '_simple_change'], self.record['event_' +
                                                               event_type +
                                                               '_simple_change']])

            self.df.loc[self.df.index[loc], 'event_time'] = True

    def make_timesplit_dist(self, assert_on: bool = False):

        if (len(self.time_splits) == 0):
            self.time_splits = [(0, 24.1)]

        ts_by_hour_min = pd.Series(self.df.index.values)
        ts_by_hour_min = ts_by_hour_min.dt.hour.values + \
            ts_by_hour_min.dt.minute.values / 60

        tem_loc = np.array([], dtype=np.int64)

        if not type(self).reuse_eventdata_4timesplit:
            event_loc = np.where(self.df.loc[:, 'event_time'].values)[0]
        else:
            event_loc = np.array([])

        for i in range(len(self.time_splits)):
            split = self.time_splits[i]
            start_t, end_t = split[0], split[1]

            loc = np.where(np.logical_and(ts_by_hour_min >=
                           start_t, ts_by_hour_min < end_t))[0]

            if type(self).reuse_eventdata_4timesplit:
                pass
            else:
                loc = set(loc) - set(event_loc)
                loc = sorted(loc)
                loc = np.asarray(loc, dtype=np.int64)
                # loc may be empty. but it should be OK.

            self.record['timesep_' + str(i) + '_loc'] = loc
            tem_loc = np.concatenate([tem_loc, loc])

            if 'log_return' in self.df.columns:
                self.record['timesep_' +
                            str(i) +
                            '_log_return'] = self.df.log_return.values[loc]

            if 'simple_change' in self.df.columns:
                self.record['timesep_' +
                            str(i) +
                            '_simple_change'] = self.df.simple_change.values[loc]

        # the remaining timestamps not included in the time splits
        i = i + 1
        assert (i == len(self.time_splits))

        loc = set(np.arange(len(ts_by_hour_min))) - set(event_loc)
        loc = loc - set(tem_loc)
        loc = sorted(loc)
        loc = np.asarray(loc, dtype=np.int64)
        self.record['timesep_' + str(i) + '_loc'] = loc

        if 'log_return' in self.df.columns:
            self.record['timesep_' +
                        str(i) +
                        '_log_return'] = self.df.log_return.values[loc]

        if 'simple_change' in self.df.columns:
            self.record['timesep_' +
                        str(i) +
                        '_simple_change'] = self.df.simple_change.values[loc]

        if assert_on:
            n = 0
            for j in range(i + 1):
                n += len(self.record['timesep_' + str(j) + '_loc'])

            if type(self).reuse_eventdata_4timesplit:
                assert (n == len(self.df))
            else:
                m = len(event_loc)
                assert ((n + m) == len(self.df))

    def get_timesplit_dist(self, i, type_of_return='log_return'):

        key = f'timesep_{i}_{type_of_return}'
        if key in self.record:
            return self.record[key]
        else:
            return np.array([])

    def get_event_dist(self, event_type, type_of_return='log_return'):

        key = f'event_{event_type}_{type_of_return}'

        if key in self.record:
            return self.record[key]
        else:
            return np.array([])

    def get_timesplit_loc(self, i):
        return self.record['timesep_' + str(i) + '_loc']

    def get_event_loc(self, i):
        return self.record['event_' + str(i) + '_loc']


def load_oneday_data(list_of_files_by_day, i):
    """
    load one day FX data
    """

    file = list_of_files_by_day[i]
    tem = pd.read_csv(file)
    tem = tem.loc[tem['bid'] != 'bid', :]
    tem[['bid', 'ask']] = tem[['bid', 'ask']].astype(float)
    tem['mid'] = 0.5 * (tem.bid + tem.ask)
    tem = tem.loc[tem['mid'].notna(), :]

    tem['time'] = pd.to_datetime(tem.ts, format='%Y-%m-%d %H:%M:%S')
    tem.sort_values('time', inplace=True)

    return pd.Series(tem.time.values), tem.mid.values


def load_files(data_dir, start_date, end_date):
    files = sorted(os.listdir(data_dir))
    try:
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        dates = np.asarray([datetime.strptime(f[:10], '%Y-%m-%d')
                           for f in files])

        loc = np.where(np.logical_and(
            dates >= start_date, dates <= end_date))[0]

        idx = np.argsort(loc)
        files = [files[x] for x in loc]
        sorted_files = [files[x] for x in idx]

    except Exception as e:
        raise e

    return sorted_files


def build_index(
        model_settings: dict,
        data_dir: str,
        start_date: str,
        end_date: str,
        events_data: pd.DataFrame,
        event_date_list: List,
        store_ts: OrderedDict,
        store_index_logreturn: OrderedDict,
        store_fx: OrderedDict,
        fx_index_starting_value: Optional[float],
        **other_attributes):
    """
    index construction.
    other_attribributes: provide "load_overnight_return_in_lb = True", if also loading overnight returns in the first
    look back periods. this was not done in previous versions.
    """

    # the state variables of the algorithm: the number of days for which the
    # index has already been built
    size_store = len(store_ts)

    # model settings /part 1:
    sigma_def = model_settings['sigma_def']
    vol_annualization_factor = model_settings['vol_annualization_factor']
    build_with_logreturns = model_settings['build_with_logreturns']
    build_with_simplechanges = model_settings['build_with_simplechanges']
    transformation = model_settings['transformation']
    event_transformation = model_settings['event_transformation']

    shift_type = model_settings['shift_type']
    shift_parameter = model_settings['shift_parameter']

    if 'mean_scaling_factor' not in model_settings:
        mean_scaling_factor = 1
    elif model_settings['mean_scaling_factor'] is None:
        mean_scaling_factor = 1
    else:
        mean_scaling_factor = model_settings['mean_scaling_factor']

    # model settings /part 2 : five key parameters that are subject to
    # calibration
    threshold = model_settings['threshold']
    norm2t_dof = model_settings['norm2t_dof']
    t2norm_dof = model_settings['t2norm_dof']
    time_splits = model_settings['time_splits']
    n_lookback_days = model_settings['n_lookback_days']

    # model settings /part3 : event settings:
    adjust_events = model_settings['adjust_events']
    pre_event_timedelta = model_settings['pre_event_timedelta']
    post_event_timedelta = model_settings['post_event_timedelta']
    event_types = model_settings['event_types']

    # load FX data
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    files = sorted(os.listdir(data_dir))
    file_dates = np.asarray(
        [datetime.strptime(f[:10], '%Y-%m-%d').date() for f in files])

    past_dates = file_dates[file_dates < start_date.date()]

    lb_tick_file = load_files(data_dir,
                              past_dates[-n_lookback_days:][0].strftime('%Y-%m-%d'),
                              past_dates[-1].strftime('%Y-%m-%d'))
    lb_tick_file = [os.path.join(data_dir, i) for i in lb_tick_file]

    # tick files ranging from the start date of the first lookback period to
    # the end date of the index-run period
    tick_file = load_files(data_dir, start_date.strftime(
        '%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    tick_file = [os.path.join(data_dir, f) for f in tick_file]
    tick_file = lb_tick_file + tick_file
    # end:load data

    print('--------')
    print('lookback period', past_dates[-n_lookback_days:]
          [0].strftime('%Y-%m-%d'), past_dates[-1].strftime('%Y-%m-%d'))
    print('running period', start_date.strftime(
        '%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    print('--------')

    lb_data = dict()
    eod_fx, model_dates = [], []
    yesterday = None

    for i in tqdm(range(len(tick_file))):

        # the first lookback period : no index-running
        if i < n_lookback_days:

            # Generation time, spot
            ts, fx_spot = load_oneday_data(tick_file, i)
            assert (np.sum(np.isnan(fx_spot)) == 0)

            # shifts
            fx_shifts = calculate_shift(
                fx_spot[:-1], fx_spot[1:], shift_type, shift_parameter)

            # Test whether events happen
            this_date = ts.dt.date[0].strftime('%Y-%m-%d')
            event = np.any(np.isin(event_date_list, this_date))
            event = event and adjust_events
            if event:
                # Generate event list
                event_list = events_data.loc[events_data['date'] == this_date, :].reset_index(
                    drop=True)
            else:
                event_list = []

            if ('load_overnight_return_in_lb' in other_attributes):
                load_overnight_return_in_lb = other_attributes['load_overnight_return_in_lb']
            else:
                load_overnight_return_in_lb = False

            if load_overnight_return_in_lb:
                if i == 0:
                    last_fx_file_before_lb = load_files(data_dir,
                                                        past_dates[-(1 + n_lookback_days):][0].strftime('%Y-%m-%d'),
                                                        past_dates[-(1 + n_lookback_days):][0].strftime('%Y-%m-%d'))
                    last_fx_file_before_lb = [os.path.join(
                        data_dir, f) for f in last_fx_file_before_lb]
                    _, fx_spot_before_lb = load_oneday_data(
                        last_fx_file_before_lb, 0)
                    eod_fx_before_lb = fx_spot_before_lb[-1]

                    overnight_return = calculate_shift(
                        eod_fx_before_lb, fx_spot[0], shift_type, shift_parameter)

                    ###########################################################

                    fx_shifts = np.insert(fx_shifts, 0, overnight_return)

                    intraday_sep = IntradaySeparation(
                        ts, fx_shifts, np.array(
                            []), time_splits, event, event_list)

                else:

                    overnight_return = calculate_shift(
                        eod_fx[-1], fx_spot[0], shift_type, shift_parameter)
                    ###########################################################

                    fx_shifts = np.insert(fx_shifts, 0, overnight_return)

                    intraday_sep = IntradaySeparation(
                        ts, fx_shifts, np.array(
                            []), time_splits, event, event_list)

            else:
                intraday_sep = IntradaySeparation(
                    ts, np.insert(
                        fx_shifts, 0, 0), np.array(
                        []), time_splits, event, event_list)

            intraday_sep.make_event_dist(
                pre_event_timedelta, post_event_timedelta, event_types)
            intraday_sep.make_timesplit_dist(assert_on=True)

            lb_data[i] = intraday_sep
            model_dates.append(ts.dt.date[0])
            eod_fx.append(fx_spot[-1])

            if (i == (n_lookback_days - 1)) and (size_store == 0):
                if fx_index_starting_value is None:
                    # set the starting value of the index to be the end-of-day
                    # FX before index starts running
                    fx_index_starting_value = eod_fx[-1]

            if i > 0:
                # ensure that dates are ordered correctly
                assert (ts.dt.date[0] > yesterday)

            yesterday = ts.dt.date[0]

            #### end of initial lookback period ############

        else:

            # start building the index

            ts, fx_spot = load_oneday_data(tick_file, i)
            assert (np.sum(np.isnan(fx_spot)) == 0)

            fx_shifts = calculate_shift(
                fx_spot[:-1], fx_spot[1:], shift_type, shift_parameter)

            # add the overnight return

            overnight_return = calculate_shift(
                eod_fx[-1], fx_spot[0], shift_type, shift_parameter)

            fx_shifts = np.insert(fx_shifts, 0, overnight_return)

            # ensure that dates are ordered correctly
            assert (ts.dt.date[0] > yesterday)

            # Test whether there is any event
            this_date = ts.dt.date[0].strftime('%Y-%m-%d')
            event = np.any(np.isin(event_date_list, this_date))
            event = event and adjust_events
            if event:
                event_list = events_data.loc[events_data['date'] == this_date, :].reset_index(
                    drop=True)
            else:
                event_list = []

            if build_with_logreturns and build_with_simplechanges:
                raise NotImplementedError

            elif build_with_logreturns:

                intraday_sep = IntradaySeparation(
                    ts, fx_shifts, np.array(
                        []), time_splits, event, event_list)

                intraday_sep.make_event_dist(
                    pre_event_timedelta, post_event_timedelta, event_types)
                intraday_sep.make_timesplit_dist(assert_on=True)

                index_log_returns = np.zeros(len(fx_shifts))

                for k in range(len(time_splits) + 1):

                    loc = intraday_sep.get_timesplit_loc(k)
                    fx_returns_k = fx_shifts[loc]
                    lb_returns = np.concatenate([lb_data[i - (j + 1)].get_timesplit_dist(
                        k, 'log_return') for j in range(n_lookback_days - 1, -1, -1)])

                    assert (len(lb_returns) > 0)

                    if (transformation == mixing_t2norm_norm2t):
                        brownian, mu, alpha, beta = transformation(
                            fx_returns_k, lb_returns, t2norm_dof, norm2t_dof, threshold)

                    else:
                        raise NotImplementedError

                    # mu = 0
                    index_log_returns[loc] = mean_scaling_factor * mu + sigma_def / \
                        vol_annualization_factor * (brownian - beta) / alpha

                for k in range(len(event_list)):

                    loc = intraday_sep.get_event_loc(k)
                    fx_returns_k = fx_shifts[loc]

                    event_type = event_list.easy[k]
                    lb_event_returns = np.concatenate([lb_data[i - (j + 1)].get_event_dist(event_type) for j in
                                                       range(n_lookback_days - 1, -1, -1)])

                    if len(lb_event_returns) == 0:
                        assert (IntradaySeparation.reuse_eventdata_4timesplit)
                        continue

                    if (event_transformation == mixing_t2norm_norm2t):
                        brownian, mu, alpha, beta = event_transformation(
                            fx_returns_k, lb_event_returns, t2norm_dof, norm2t_dof, threshold)

                    elif (event_transformation == t2norm_transform):
                        dof = t2norm_dof
                        brownian, mu, alpha, beta = event_transformation(
                            fx_returns_k, lb_event_returns, dof)

                    else:
                        raise NotImplementedError

                    # mu = 0
                    index_log_returns[loc] = mean_scaling_factor * mu + sigma_def / \
                        vol_annualization_factor * (brownian - beta) / alpha

                store_index_logreturn[size_store + i -
                                      n_lookback_days] = index_log_returns

            elif build_with_simplechanges:
                raise NotImplementedError('to be added')

            # finish running the index

            lb_data[i] = intraday_sep
            model_dates.append(ts.dt.date[0])
            eod_fx.append(fx_spot[-1])

            # release data memory(that will not be used). it can build a longer
            # history for the index

            if (i - n_lookback_days) in lb_data:
                del lb_data[(i - n_lookback_days)]

            store_fx[size_store + i - n_lookback_days] = fx_spot
            store_ts[size_store + i - n_lookback_days] = ts

            yesterday = ts.dt.date[0]
        # end of buiding index
         # end of for i in range(len(tick_file))

    assert (len(store_ts) == (size_store + (len(tick_file) - n_lookback_days)))
    # end of def one_period_run(self, model_settings: dict, data_dir: str,
    # start_date: str, end_date: str)

    return (store_ts, store_index_logreturn, store_fx, fx_index_starting_value)


def calculate_statistics(
        store_ts: OrderedDict,
        store_index_logreturn: OrderedDict,
        store_fx: OrderedDict,
        fx_index_starting_value: float,
        events_data: pd.DataFrame,
        model_settings: dict,
        output_dir: Optional[str],
        save_index: bool,
        save_mode: str):
    post_event_timedelta = model_settings['post_event_timedelta']
    vol_annualization_factor = model_settings['vol_annualization_factor']
    ccy = model_settings['ccy']

    ts = np.concatenate([store_ts[m] for m in store_ts.keys()])
    fx = np.concatenate([store_fx[m] for m in store_fx.keys()])

    all_logreturn = np.concatenate(
        [store_index_logreturn[m] for m in store_index_logreturn.keys()])

    fx_index = np.exp(all_logreturn.cumsum()) * \
        100000  # fx_index_starting_value #

    store_fx.clear()
    store_ts.clear()
    store_index_logreturn.clear()

    start_date = pd.Timestamp(ts[0]).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(ts[-1]).strftime("%Y-%m-%d")

    df_results = pd.DataFrame(data=np.vstack(
        (fx_index, fx)).T, index=ts, columns=['Index', 'FX'])

    del ts, fx, fx_index

    if (output_dir is not None) and save_index:

        assert (fx_index_starting_value is not None)

        if save_mode == 'pkl':
            df_results.to_pickle(
                os.path.join(
                    output_dir,
                    f'Index_{ccy}_from_{start_date}_to_{end_date}.pkl'))
        elif save_mode == 'csv':
            df_results.to_csv(
                os.path.join(
                    output_dir,
                    f'Index_{ccy}_from_{start_date}_to_{end_date}.csv'))
        else:
            df_results.to_pickle(
                os.path.join(
                    output_dir,
                    f'Index_{ccy}_from_{start_date}_to_{end_date}.pkl'))

    stats = {}

    sigma_def = model_settings['sigma_def']

    stats['sigma_def'] = sigma_def

    # resampled_df = df_results.resample('1D').last().dropna()
    # daily_index_returns = resampled_df.loc[:, 'Index'].map(np.log).diff(periods=1).iloc[1:].values
    # daily_fx_returns = resampled_df.loc[:, 'FX'].map(np.log).diff(periods=1).iloc[1:].values

    # stats["Autocorr of (daily) Index returns"]    = round(ss.pearsonr(daily_index_returns[1:], daily_index_returns[:-1])[0], 4)
    # stats["Autocorr of (daily) FX returns"]       = round(ss.pearsonr(daily_fx_returns[1:], daily_fx_returns[:-1])[0], 4)
    # stats["Pearson Corr btw the (daily) returns"] = round(ss.pearsonr(daily_fx_returns, daily_index_returns)[0], 4)

    # del daily_index_returns, daily_fx_returns

    for freq in ['1H', '1D']:

        resampled_df = df_results.resample(freq).last().dropna()
        resampled_index_returns = resampled_df.loc[:, 'Index'].map(
            np.log).diff(periods=1).iloc[1:].values
        resampled_fx_returns = resampled_df.loc[:, 'FX'].map(
            np.log).diff(periods=1).iloc[1:].values

        stats[f"Autocorr of ({freq}) Index returns"] = round(ss.pearsonr(
            resampled_index_returns[1:], resampled_index_returns[:-1])[0], 4)
        # stats[f"Vol of ({freq}) Index Returns"] = round(np.std(resampled_index_returns), 4)

        stats[f"Autocorr of ({freq}) FX returns"] = round(ss.pearsonr(
            resampled_fx_returns[1:], resampled_fx_returns[:-1])[0], 4)
        # stats[f"Vol of ({freq}) FX Returns"] = round(np.std(resampled_fx_returns), 4)

        # stats[f"Pearson Corr btw the ({freq}) returns"] = round(ss.pearsonr(resampled_fx_returns, resampled_index_returns)[0], 4)
        # stats[f"Kendall Tau Corr btw the ({freq}) returns"] = round(ss.kendalltau(resampled_fx_returns, resampled_index_returns)[0], 4)

    index_returns = df_results.loc[:, 'Index'].map(
        np.log).diff(periods=1).iloc[1:]
    index_returns.index = df_results.index[1:]

    std_index_returns = np.std(index_returns.values)

    stats["Annualized Vol of (one-second) Index returns"] = round(
        std_index_returns * vol_annualization_factor, 4)

    intraday_vol = index_returns.groupby(
        [index_returns.index.date, index_returns.index.hour]).std().values * vol_annualization_factor
    intraday_vol = intraday_vol[~np.isnan(intraday_vol)]

    stats['Intraday_60min_Vol_0.1%quantile'] = round(
        np.quantile(intraday_vol, 0.001), 4)
    # stats['Intraday_60min_Vol_1%quantile']    = round(np.quantile(intraday_vol, 0.01), 4)
    # stats['Intraday_60min_Vol_2.5%quantile']  = round(np.quantile(intraday_vol, 0.025), 4)
    stats['Intraday_60min_Vol_Average'] = round(np.mean(intraday_vol), 4)
    # stats['Intraday_60min_Vol_97.5%quantile'] = round(np.quantile(intraday_vol, 0.975), 4)
    # stats['Intraday_60min_Vol_99%quantile']   = round(np.quantile(intraday_vol, 0.99), 4)
    stats['Intraday_60min_Vol_99.9%quantile'] = round(
        np.quantile(intraday_vol, 0.999), 4)

    # stats['Intraday_60min_Vol_99% bandwidth'] = round(np.quantile(intraday_vol, 0.99)- np.quantile(intraday_vol, 0.01), 4)
    stats['Intraday_60min_Vol_99.9% bandwidth'] = round(np.quantile(
        intraday_vol, 0.999) - np.quantile(intraday_vol, 0.001), 4)

    Fx_returns = df_results.loc[:, 'FX'].map(np.log).diff(periods=1).iloc[1:]
    Fx_returns.index = df_results.index[1:]

    intraday_vol_Fx = Fx_returns.groupby(
        [Fx_returns.index.date, Fx_returns.index.hour]).std().values * vol_annualization_factor
    intraday_vol_Fx = intraday_vol_Fx[~np.isnan(intraday_vol_Fx)]

    stats['Intraday_60min_Vol_0.1%quantile_Fx'] = round(
        np.quantile(intraday_vol_Fx, 0.001), 4)
    stats['Intraday_60min_Vol_99.9%quantile_Fx'] = round(
        np.quantile(intraday_vol_Fx, 0.999), 4)
    stats['Intraday_60min_Vol_99.9%_Fx bandwidth'] = round(np.quantile(
        intraday_vol_Fx, 0.999), 4) - round(np.quantile(intraday_vol_Fx, 0.001), 4)

    #### event vol ####

    if True:

        ts_start = df_results.index[0]
        ts_end = df_results.index[-1]

        if len(events_data) > 0:
            included_events = events_data.loc[(events_data['time'] >= ts_start) & (
                events_data['time'] <= ts_end), :].reset_index(drop=True)
        else:
            included_events = pd.DataFrame()

        if len(included_events) > 0:
            index_event_vol = []
            fx_event_vol = []

            n_mins = post_event_timedelta / timedelta(minutes=1)

            stat_pre_event_timedelta = timedelta(minutes=-1 * n_mins)
            stat_post_event_timedelta = timedelta(minutes=1 * n_mins)

            for k in range(len(included_events)):
                event_time = included_events.loc[k, 'time']
                event_results = df_results.loc[(df_results.index >= (event_time + stat_pre_event_timedelta)) & (
                    df_results.index <= (event_time + stat_post_event_timedelta)),]

                index_event = event_results.loc[:, 'Index'].values
                index_event_returns = np.log(index_event)[
                    1:] - np.log(index_event)[:-1]
                index_event_vol.append(
                    np.std(index_event_returns) * vol_annualization_factor)

                fx_event = event_results.loc[:, 'FX'].values
                fx_event_returns = np.log(fx_event)[1:] - np.log(fx_event)[:-1]
                fx_event_vol.append(np.std(fx_event_returns)
                                    * vol_annualization_factor)

            index_event_vol = np.array(index_event_vol)
            index_event_vol = index_event_vol[~np.isnan(index_event_vol)]

            stats['IndexVol_Around_Events_0.1%quantile'] = round(
                np.quantile(index_event_vol, 0.001), 4)
            # stats['IndexVol_Around_Events_1%quantile']    = round(np.quantile(index_event_vol, 0.01), 4)
            # stats['IndexVol_Around_Events_2.5%quantile']  = round(np.quantile(index_event_vol, 0.025), 4)
            stats['Average_IndexVol_Around_Events'] = round(
                np.mean(index_event_vol), 4)
            # stats['IndexVol_Around_Events_97.5%quantile'] = round(np.quantile(index_event_vol, 0.975), 4)
            # stats['IndexVol_Around_Events_99%quantile']   = round(np.quantile(index_event_vol, 0.99), 4)
            stats['IndexVol_Around_Events_99.9%quantile'] = round(
                np.quantile(index_event_vol, 0.999), 4)

            # stats['IndexVol_Around_Events_99% bandwidth'] = round(np.quantile(index_event_vol, 0.99) - np.quantile(index_event_vol, 0.01), 4)
            stats['IndexVol_Around_Events_99.9% bandwidth'] = round(np.quantile(
                index_event_vol, 0.999) - np.quantile(index_event_vol, 0.001), 4)

            fx_event_vol = np.array(fx_event_vol)
            fx_event_vol = fx_event_vol[~np.isnan(fx_event_vol)]

            stats['FXVol_Around_Events_0.1%quantile'] = round(
                np.quantile(fx_event_vol, 0.001), 4)
            # stats['FXVol_Around_Events_1%quantile'] = round(np.quantile(fx_event_vol, 0.01), 4)
            # stats['FXVol_Around_Events_2.5%quantile'] = round(np.quantile(fx_event_vol, 0.025), 4)
            # stats['Average_FXVol_Around_Events'] = round(np.mean(fx_event_vol), 4)
            # stats['FXVol_Around_Events_97.5%quantile'] = round(np.quantile(fx_event_vol, 0.975), 4)
            # stats['FXVol_Around_Events_99%quantile'] = round(np.quantile(fx_event_vol, 0.99), 4)
            stats['FXVol_Around_Events_99.9%quantile'] = round(
                np.quantile(fx_event_vol, 0.999), 4)
            stats['FXVol_Around_Events_99.9% bandwidth'] = round(
                np.quantile(fx_event_vol, 0.999) - np.quantile(fx_event_vol, 0.001), 4)

        #### event vol ####

    if output_dir is not None:
        pd.Series(stats).to_csv(os.path.join(
            output_dir, f'stat_{ccy}_from_{start_date}_to_{end_date}.csv'))

    return stats


def wrapper_add_statistics(func_build_index):
    """
    wrap up the index construction function and add model statistics
    """

    @functools.wraps(func_build_index)
    def wrapper_build_index(*args, **kwargs):
        """
        wrapper of the build_index function so that it calcualtes model statistics of the index
        """

        store_ts, store_index_logreturn, store_fx, fx_index_starting_value = func_build_index(
            *args, **kwargs)

        # calculate model statistics after running the index
        # note that model_settings has to passed to func_build_index as a
        # keyword parameter
        model_settings = kwargs['model_settings']
        events_data = kwargs['events_data']
        output_dir = kwargs['output_dir']
        save_index = kwargs['save_index']
        save_mode = kwargs['save_mode']

        index_stat = calculate_statistics(
            store_ts,
            store_index_logreturn,
            store_fx,
            fx_index_starting_value,
            events_data,
            model_settings,
            output_dir,
            save_index,
            save_mode)

        return index_stat

    return wrapper_build_index


class IndexEngine:
    """
    the class that builds the index
    """

    def __init__(self):

        # store fx, index,  timestamp, and initial value  --  state variables
        # for multi-period runs
        self.store_fx = OrderedDict()
        self.store_index_logreturn = OrderedDict()
        self.store_ts = OrderedDict()
        self.fx_index_starting_value = 100000  # None

    def load(
            self,
            model_settings: dict,
            fx_data_dir: str,
            events_data: pd.DataFrame,
            event_date_list: List):

        self.model_settings = copy.deepcopy(model_settings)
        self.data_dir = fx_data_dir
        self.events_data = events_data
        self.event_date_list = event_date_list

        # update relevant event types
        # we filter event_types against the events in the events_data
        # keep only event_types that are present in the data

        event_types = copy.deepcopy(model_settings['event_types'])

        if len(self.events_data) > 0:
            event_types_in_data = set(self.events_data['easy'])

            for t in model_settings['event_types']:
                if t in event_types_in_data:
                    pass
                else:
                    event_types.remove(t)
        else:
            event_types = []

        self.model_settings['event_types'] = event_types

    def one_period_run(
            self,
            start_date: str,
            end_date: str,
            output_dir: Optional[str],
            save_index=True,
            save_mode='pkl',
            func_build_index=build_index):

        build_index = wrapper_add_statistics(func_build_index)

        index_stat = build_index(
            model_settings=self.model_settings,
            data_dir=self.data_dir,
            start_date=start_date,
            end_date=end_date,
            events_data=self.events_data,
            event_date_list=self.event_date_list,
            store_ts=self.store_ts,
            store_index_logreturn=self.store_index_logreturn,
            store_fx=self.store_fx,
            fx_index_starting_value=self.fx_index_starting_value,
            output_dir=output_dir,
            save_index=save_index,
            save_mode=save_mode)

        return index_stat

    def multi_period_runs(self, start_date: str,
                          end_date: str,
                          run_periods: Union[PeriodIndex, List],
                          list_model_settings: List[dict],
                          output_dir: Optional[str],
                          save_index: bool,
                          save_mode: str,
                          cache_result_dir: Optional[str] = None,
                          calibration_period: Optional[str] = None,
                          new_calibration: bool = False,
                          parallel=False):

        assert (len(run_periods) == len(list_model_settings))

        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

        if not parallel:

            for i in range(len(run_periods)):
                p = run_periods[i]

                p_start_date = max(p.start_time, start_date.date())
                p_end_date = min(p.end_time, end_date.date())

                p_start_date = datetime.strftime(p_start_date, '%Y-%m-%d')
                p_end_date = datetime.strftime(p_end_date, '%Y-%m-%d')

                model_settings = list_model_settings[i]

                if i == 0:
                    results = build_index(
                        model_settings,
                        data_dir=self.data_dir,
                        start_date=p_start_date,
                        end_date=p_end_date,
                        events_data=self.events_data,
                        event_date_list=self.event_date_list,
                        store_ts=self.store_ts,
                        store_index_logreturn=self.store_index_logreturn,
                        store_fx=self.store_fx,
                        fx_index_starting_value=self.fx_index_starting_value)

                else:
                    results = build_index(
                        model_settings,
                        data_dir=self.data_dir,
                        start_date=p_start_date,
                        end_date=p_end_date,
                        events_data=self.events_data,
                        event_date_list=self.event_date_list,
                        store_ts=self.store_ts,
                        store_index_logreturn=self.store_index_logreturn,
                        store_fx=self.store_fx,
                        fx_index_starting_value=self.fx_index_starting_value,
                        load_overnight_return_in_lb=True)

                # have to assign the returned value to
                # self.fx_index_starting_value (because fx_index_starting_value
                # is rebind in build_index )
                if i == 0:
                    self.fx_index_starting_value = results[-1]
                    print(
                        f'start value of index {self.fx_index_starting_value}')

        else:
            raise NotImplementedError

        self.output(output_dir, save_index, save_mode)

    def output(
            self,
            output_dir: str,
            save_index: bool = False,
            save_mode: str = 'pkl') -> dict:

        stat = calculate_statistics(self.store_ts,
                                    self.store_index_logreturn,
                                    self.store_fx,
                                    self.fx_index_starting_value,
                                    self.events_data,
                                    self.model_settings,
                                    output_dir,
                                    save_index,
                                    save_mode
                                    )

        return stat


if __name__ == '__main__':

    ccy = 'GBPUSD'
    sigma_def = 1

    start_date = '2016-02-01'
    end_date = '2016-12-31'

    # ### data dir  ####
    # if ccy == 'USDJPY':
    #     fx_input_dir = r'C:\Users\smile\OneDrive - FF Quant Advisory B.V\Documents\01 Deriv.com\0. Index\MarketData\USDJPY\tick_2012_2021'
    # elif ccy == 'AUDUSD':
    #     fx_input_dir = r'C:\Users\smile\OneDrive - FF Quant Advisory B.V\Documents\01 Deriv.com\0. Index\MarketData\AUDUSD\AUDUSD_daily'
    # elif ccy == 'EURUSD':
    #     fx_input_dir = r'C:\Users\smile\OneDrive - FF Quant Advisory B.V\Documents\01 Deriv.com\0. Index\MarketData\EURUSD\EURUSD_daily'
    # elif ccy == 'GBPUSD':
    #     fx_input_dir = r'C:\Users\smile\OneDrive - FF Quant Advisory B.V\Documents\01 Deriv.com\0. Index\MarketData\GBPUSD\tick_2016_2020'
    # else:
    #     raise Exception(f"CCY not supported: {ccy} ")
    if ccy not in ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD", "USDCHF"]:
        raise NotImplementedError(f"CCY not supported: {ccy} ")

    ### data dir  ####
    fx_input_dir = r'/Users/chunkiat/Documents/Python/Main/R & D/Normalised_FX_Index/{}_daily'.format(
        ccy)

    # event_input_file = r'C:\Users\smile\OneDrive - FF Quant Advisory B.V\Documents\01 Deriv.com\0. Index\MarketData\news_list_full_2016_2021.csv'
    event_input_file = r'/Users/chunkiat/Documents/Python/Main/R & D/Normalised_FX_Index/news_list_full_2016_2021.csv'
    ######

    ####### events data ##########
    symbols = ["USD", "GBP", "JPY", "AUD", "NZD", "EUR", "CAD", "CHF"]

    event_ts_col = 'ts'  # old version :'trans_release_time'
    event_types = ['high', 'medium', 'low']

    events_data = pd.read_csv(event_input_file,
                              parse_dates=[event_ts_col])
    events_data = events_data.loc[:, [
        'event_name', 'symbol', event_ts_col, 'impact']]

    events_data = events_data.loc[events_data['symbol'].isin(symbols), :]

    # events_data = events_data.loc[(events_data['symbol'] == ccy[:3]) | (
    #         events_data['symbol'] == ccy[3:]), :]

    events_data['easy'] = np.where(
        events_data['impact'] < 3, 'low', np.where(
            events_data['impact'] < 5, 'medium', 'high'))

    events_data['time'] = events_data[event_ts_col]
    events_data['date'] = events_data['time'].dt.strftime('%Y-%m-%d')

    events_data = events_data.loc[events_data['time'].dt.year.between(
        2016, 2021), :]
    events_data = events_data.drop_duplicates(
        ['event_name', 'symbol', 'impact', 'time'])
    events_data['easy_label'] = np.where(
        events_data['easy'] == 'low', 0, np.where(
            events_data['easy'] == 'medium', 1, 2))

    events_data = events_data.loc[events_data['easy'].isin(event_types),]

    events_data = events_data.sort_values(by=['time', 'easy_label'])
    events_data = events_data.drop_duplicates(
        ['time', 'easy_label'], keep='last')
    print(events_data["symbol"].unique())
    print(events_data["easy"].unique())

    event_date_list = list(events_data['date'].unique())  # event date
    event2 = events_data

    #####################################

    #### model settings ###

    model_settings = {}
    model_settings['ccy'] = ccy
    model_settings['sigma_def'] = sigma_def
    model_settings['vol_annualization_factor'] = np.sqrt(252 * 86400)

    model_settings['adjust_events'] = True
    model_settings['event_types'] = event_types
    model_settings['pre_event_timedelta'] = timedelta(minutes=-30)
    model_settings['post_event_timedelta'] = timedelta(minutes=30)

    model_settings['build_with_logreturns'] = True

    model_settings['build_with_simplechanges'] = False
    model_settings['transformation'] = mixing_t2norm_norm2t
    model_settings['event_transformation'] = t2norm_transform

    # model_settings['shift_type'] = np.arcsinh  # shift using inverse
    # hyperbolic sine
    model_settings['shift_type'] = np.log     # shift using log

    model_settings['shift_parameter'] = 1.0
    # use 1.0 for look-back mean, use 0 for zero-mean
    model_settings['mean_scaling_factor'] = 1.0

    threshold = 5
    norm2t_dof = 10
    t2norm_dof = 10
    n_lookback_days = 10

    # [(0, 2), (12, 16), (21, 23)], # 4 splits
    # [(0, 2), (2, 6), (6, 12), (12, 16), (21, 23)], # 6splits
    # [(0, 2), (6.5, 11.5), (11.5, 16.5), (21, 21.3), (21.3, 21.8), (21.8, 22.05)] #7splits

    # time_splits = [(0, 2), (12, 16), (21, 23)]
    time_splits = [(0, 2), (2, 6), (6, 12), (12, 16), (21, 23)]

    model_settings['threshold'] = threshold
    model_settings['norm2t_dof'] = norm2t_dof
    model_settings['t2norm_dof'] = t2norm_dof
    model_settings['n_lookback_days'] = n_lookback_days
    model_settings['time_splits'] = time_splits

    run_id = f'./{ccy}_vol{sigma_def}_test_v10/{start_date}_{end_date}_n2t{norm2t_dof}_t2n{t2norm_dof}_threshold{threshold}_lb{n_lookback_days}_ts{len(time_splits)+1}/'

    output_dir = os.path.abspath(run_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # initialize the engine
    index_engine = IndexEngine()
    index_engine.load(model_settings, fx_input_dir,
                      events_data, event_date_list)

    index_stat = index_engine.one_period_run(start_date, end_date, output_dir,
                                             save_index=True)
