
import pandas as pd
import numpy as np
import os
from datetime import timedelta, datetime
import time
import functools
import copy

from fxderived import IndexEngine, mixing_t2norm_norm2t, t2norm_transform


ccy = 'USDCHF'
sigma_def = 0.1

using_fixed_parameter = True


### data dir  ####

fx_input_dir = r'frx' + ccy + '/daily'  # ../usdchf/USDCHF/daily'# #.
# '../eurusd/EURUSD/news_list_full_2016_2021.csv'#economic_events_parsed.csv'#event_calendar_final.csv' #'
event_input_file = 'parsed_calendar_4.csv'


####### events data ##########
symbols = ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CAD", "CHF"]

event_ts_col = 'ts'  # old version :'trans_release_time'
event_types = ['high', 'medium', 'low']

events_data = pd.read_csv(event_input_file,
                          parse_dates=[event_ts_col])
# events_data = events_data.loc[:, ['event_name', 'symbol', event_ts_col, 'impact']]

# events_data = events_data.loc[events_data['symbol'].isin(symbols), :]

events_data = events_data.loc[(events_data['symbol'] == ccy[:3]) | (
    events_data['symbol'] == ccy[3:]), :]

events_data['easy'] = np.where(
    events_data['impact'] < 3, 'low', np.where(
        events_data['impact'] < 5, 'medium', 'high'))

events_data['time'] = pd.to_datetime(events_data[event_ts_col], utc=True)
# events_data['time'].dt.strftime('%Y-%m-%d')
events_data['date'] = events_data['time'].dt.date

events_data = events_data.loc[events_data['time'].dt.year.between(
    2016, 2022), :]
# events_data = events_data.drop_duplicates(['event_name', 'symbol', 'impact', 'time'])
events_data['easy_label'] = np.where(
    events_data['easy'] == 'low', 0, np.where(
        events_data['easy'] == 'medium', 1, 2))

events_data = events_data.loc[events_data['easy'].isin(event_types),]

events_data = events_data.sort_values(by=['time', 'easy_label'])
events_data = events_data.drop_duplicates(['time', 'easy_label'], keep='last')
print(events_data["symbol"].unique())
print(events_data["easy"].unique())

event_date_list = list(events_data['date'].unique())  # event date
event2 = events_data


if using_fixed_parameter:
    #### model settings ###
    for threshold in [5]:
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

        model_settings['threshold'] = threshold
        model_settings['norm2t_dof'] = 10
        model_settings['t2norm_dof'] = 10
        model_settings['n_lookback_days'] = 10
        # [(0, 2), (2, 6), (6, 12), (12, 16), (21, 23)]
        model_settings['time_splits'] = [(0, 2), (12, 16), (21, 23)]

        model_settings['shift_type'] = np.log  # arcsinh     # shift using log
        model_settings['shift_parameter'] = 1.0

        run_id = f'{ccy}-sigma{sigma_def}-{model_settings["threshold"]}-{model_settings["norm2t_dof"]}-{model_settings["t2norm_dof"]}-{model_settings["n_lookback_days"]}-{len(model_settings["time_splits"])+1}'

        output_dir = os.path.normpath(
            f'./output/BM/LAMBDA=0, vol=0.1, r=0.04, d=0, eta=0.05/{run_id}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # initialize the engine
        index_engine = IndexEngine()
        index_engine.load(model_settings, fx_input_dir,
                          events_data, event_date_list)

        start_date = '2022-02-01'
        end_date = '2022-04-30'

        # use this to trigger a one-period run
        # index_stat = index_engine.one_period_run(start_date, end_date, output_dir)
        index_engine.one_period_run(start_date, end_date, output_dir)


else:  # using calibrated parameters

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

    model_settings['shift_type'] = np.log     # shift using log
    model_settings['shift_parameter'] = 1.0

    # initialize the engine
    index_engine = IndexEngine()
    index_engine.load(model_settings, fx_input_dir,
                      events_data, event_date_list)

    start_date = '2020-02-01'
    end_date = '2020-12-31'

    run_freq = 'M'
    run_periods = pd.period_range(start_date, end_date, freq=run_freq)

    list_model_settings = []

    calibration_results = pd.read_csv(
        f'output/{ccy}_fullrun_calibration_zero_mean.csv')

    for i in range(len(run_periods)):
        # load parameters calibrated from the previous quarter
        threshold, t2norm_dof, norm2t_dof, n_lookback_days, time_splits = calibration_results.loc[i, [
            'threshold', 't2norm_dof', 'norm2t_dof', 'n_lookback_days', 'time_splits']]
        time_splits = eval(time_splits)

        updated_model_settings = copy.deepcopy(model_settings)
        updated_model_settings['threshold'] = threshold
        updated_model_settings['norm2t_dof'] = norm2t_dof
        updated_model_settings['t2norm_dof'] = t2norm_dof
        updated_model_settings['n_lookback_days'] = n_lookback_days
        updated_model_settings['time_splits'] = time_splits
        list_model_settings.append(updated_model_settings)

    run_id = f'derived_{ccy}-sigma{sigma_def}_long_run'

    output_dir = os.path.normpath(f'./output/{run_id}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    index_engine.multi_period_runs(
        start_date,
        end_date,
        run_periods,
        list_model_settings,
        output_dir=output_dir,
        save_index=True,
        save_mode='pkl')

main_end_t = time.time()
# print(f'total time : {(main_end_t - main_start_t) / 60} minutes')
