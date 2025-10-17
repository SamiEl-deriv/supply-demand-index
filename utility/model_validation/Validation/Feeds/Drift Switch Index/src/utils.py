import matplotlib.pyplot as plt
from time import perf_counter_ns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Any
from functools import wraps
import pandas as pd
from pandas.errors import ParserError
import numpy as np
import os.path as path

import yaml

def plot_feeds(df, feeds : tuple[list], titles : tuple[list], yaxes : tuple[str], sup_title, start, end, height=400):
    feeds_tuple = feeds if isinstance(feeds, tuple) else (feeds,)
    titles_tuple = feeds if isinstance(titles, tuple) else (titles,)
    yaxes_tuple = yaxes if isinstance(yaxes, tuple) else (yaxes,)

    if not len(feeds_tuple) == len(titles_tuple) == len(yaxes_tuple):
        raise ValueError("feeds, titles, yaxis tuples not all the same length")
    
    nrows = len(feeds_tuple)
    fig = make_subplots(rows=nrows, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    
    for i, (name_list, title_list, yaxis) in enumerate(zip(feeds_tuple, titles_tuple, yaxes_tuple)):
        for name, title in zip(name_list, title_list):
            fig.add_trace(go.Scatter(x=df.index[start:end], y=df[name][start:end], name=title), row=i+1, col=1)
        fig.update_yaxes(title_text=yaxis, row=i+1, col=1)

    fig.update_layout(title=dict(text=sup_title), autosize=True, height=400 * nrows,)
    fig.show()

def legend_combiner(*axes : List[plt.Axes]):
    handles = []
    axes = []
    for axis in axes:
        handles_temp, labels_temp = axis.get_legend_handles_labels()
        handles = handles + handles_temp
        labels = labels + labels_temp
    axes[0].legend(handles, labels)

def timer(subject = ''):
    """
    Decorator for tracking elapsed time

    To use, put @timer before the intended function declaration:

    @timer
    def add(x,y):
        return x + y

    >>> print(add(1,2))

    Time elapsed: 1 µs, 257 ns
    3
    """
    def timer_decorator(f):
        @wraps(f)
        def wrap(*args, **kwargs):
            start = perf_counter_ns()
            result = f(*args, **kwargs)
            stop = perf_counter_ns()
            
            left = stop - start

            # Decompose time
            nanos = left % 1000
            left = (left - nanos) // 1000

            micros = left % 1000
            left = (left - micros) // 1000

            millis = left % 1000
            left = (left - millis) // 1000

            seconds = left % 60
            mins = (left - seconds) // 60

            # Construct string
            string = f"Time elapsed"
            if subject:
                string += f' for {subject}'
            string += ': ' 
            start_string = False
            for name, time in zip(['mins', 's', 'ms', 'µs', 'ns'], 
                                [mins, seconds, millis, micros, nanos]):
                if time != 0 and not start_string:
                    start_string = True

                if start_string:
                    string += f"{time} {name}, "

            print(string[:-2])
            return result
        return wrap
    return timer_decorator

def write_yaml(filepath : str, data : Any):
    """
    Writes data to a yaml file

    Arguments
    ---------
    filepath : str
        The filepath to write to
    data : Any
        The data to write to the YAML file
    """
    with open(filepath, "w") as yaml_file:
            yaml.dump(data, yaml_file)

def read_yaml(filepath):
    """
    Reads data from a yaml file

    Arguments
    ---------
    filepath : str
        The filepath to read from

    Returns
    -------
    Any
        The yaml data as a Python Object
    """
    try:    
        with open(filepath, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except FileNotFoundError:
        print(f'File not found: {filepath}')

def read_csv_to_list_dfs(df_path : str, nrows = None):
    """
    Reads a csv containing consecutive Dataframes

    Arguments
    ---------
    df_path : str
        The filepath to the csv
    nrows : int
        The number of rows to read
        If empty, read all rows

    Returns
    -------
    list[pandas.DataFrame]
        A list of Dataframes
    """
    DSIs_raw = pd.read_csv(df_path, nrows = nrows)
    to_float = ['spot', 'bid', 'ask', 'log_returns', 'drift', 'diff']
    DSIs_raw[to_float] = DSIs_raw[to_float].astype(np.float64)
    to_int = ['Unnamed : 0', 'state']
    DSIs_raw[to_int] = DSIs_raw[to_int].astype(np.int64)
    zero_indices = list(DSIs_raw.iloc[DSIs_raw['Unnamed: 0'] == 0].index)
    zero_indices.append(len(DSIs_raw))

    zero_ranges = [(zero_indices[i], zero_indices[i+1]) for i in range(len(zero_indices) - 1)]
    DSI_list = [DSIs_raw.loc[x[0]:x[1] - 1].set_index('Unnamed : 0') for x in zero_ranges]

    return DSI_list

def write_list_dfs_to_csv(dfs : list[pd.DataFrame], df_path : str):
    """
    Writes a list of Dataframes to a csv

    Arguments
    ---------
    dfs : list[pandas.DataFrame]
        The list of Dataframes to write to the csv
    df_path : str
        The filepath to the csv
    """
    for df in dfs:
        df.to_csv(df_path, mode='a')

def read_csv_feeddb(df_path : str, nrows = None, source = "QAbox") -> pd.DataFrame:
    """
    Reads csv's generated by MT5

    Arguments
    ---------
    df_path : str
        The filepath to the csv
    nrows : int
        The number of rows to read
        If empty, read all rows

    Returns
    -------
    pandas.DataFrame
        A list of Dataframes
    """
    try:
        df = pd.read_csv(df_path, nrows=nrows)
    except ParserError:
        df = pd.read_csv(df_path, nrows=nrows, engine='python')
    

    # Parse, format and handle dates
    if source == "Metabase":
        df.loc[:,'ts'] = df['ts'].apply(lambda x : x if 'T' in x else x + "T00:00:00")
        df['ts'] = pd.to_datetime(df['ts'], format="%Y-%m-%dT%H:%M:%S")
    elif source == "QAbox":
        df = df.drop(columns=['Unnamed: 0'])
        df['ts'] = pd.to_datetime(df['ts'], format="%Y-%m-%d %H:%M:%S")
    else:
        raise ValueError("Invalid source. Try one of ['Metabase', 'QAbox']")
    df.sort_values('ts', inplace=True)

    return df





def read_csv_MT5(df_path : str, nrows = None) -> pd.DataFrame:
    """
    Reads csv's generated by MT5

    Arguments
    ---------
    df_path : str
        The filepath to the csv
    nrows : int
        The number of rows to read
        If empty, read all rows

    Returns
    -------
    pandas.DataFrame
        A list of Dataframes
    """
    DSI_raw = pd.read_csv(df_path, nrows = nrows, sep='\t', usecols=['<DATE>', '<TIME>', '<BID>', '<ASK>','<FLAGS>'])
    DSI_raw['ts'] = pd.to_datetime(DSI_raw['<DATE>'] + ' ' + DSI_raw['<TIME>'])
    DSI_raw.drop(columns=['<DATE>', '<TIME>'], inplace=True)
    DSI_raw.rename(columns=dict(zip(['<BID>', '<ASK>','<FLAGS>'], 
                                    ['bid', 'ask', 'flags'])), inplace=True)
    to_float = ['bid', 'ask']
    to_int = ['flags']
    DSI_raw[to_float] = DSI_raw[to_float].astype(np.float64)
    DSI_raw[to_int] = DSI_raw[to_int].astype(np.int64)

    cols = DSI_raw.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    DSI_raw = DSI_raw[cols]
    return DSI_raw

def write_list_dfs_to_parquet(dfs, df_path, reset=False):
    """
    Writes/appends a list of Dataframes to a parquet file

    Arguments
    ---------
    dfs : list[pandas.DataFrame]
        The list of Dataframes to write to the csv
    df_path : str
        The filepath to the parquet file
    reset : bool
        If True, overwrite file
        Else append to end of file
    """
    for df in dfs:
        temp = df.reset_index()
        if not path.isfile(df_path) or reset:
            temp.to_parquet(df_path, engine='fastparquet')
            reset = False
        else:
            temp.to_parquet(df_path, engine='fastparquet', append=True)

def read_list_dfs_from_parquet(df_path):
    """
    Reads a parquet file containing consecutive Dataframes

    Arguments
    ---------
    df_path : str
        The filepath to the csv

    Returns
    -------
    list[pandas.DataFrame]
        A list of Dataframes
    """
    df = pd.read_parquet(df_path, engine='fastparquet')
    zeroes_indices = np.append(df['index'][df['index'] == 0].index.values, df.index.values[-1] + 1)
    df = df.drop(columns='index')
    list_dfs = [df.iloc[zeroes_indices[i]:zeroes_indices[i+1],:].reset_index(drop=True) for i in range(len(zeroes_indices) - 1)]
    return list_dfs