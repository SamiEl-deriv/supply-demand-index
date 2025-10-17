import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib import gridspec as gs
from typing import Optional, Protocol
from abc import ABC, abstractmethod

DEFAULT_CMAP = plt.get_cmap('tab10')

def _renamer(arg : str, sep : str):
    """
    Utility function to rename acronyms.
    """
    code = arg.split(sep)[-1]
    match code:
        case "ML":
            return 'Trend Up'
        case "MS":
            return 'Trend Down'
        case "CL":
            return 'Rebound'
        case "CS":
            return 'Pullback'
        case "ST":
            return 'Standard'
        case code if code.isdigit():
            return code
        case _:
            return ''

class IndicatorVisualizer(ABC):
    """Abstract base class for indicator visualization."""
    
    def __init__(self, tactical: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp):
        self.tactical = tactical
        self.start = start
        self.end = end
        self.label = tactical.columns[1]
        self.underlying_name = tactical.columns[0]
        
        try:
            self.valid_start = tactical[start:end].no_lb_period[~tactical[start:end].no_lb_period].index[0]
        except IndexError:
            raise IndexError(f'Tactical Index {self.label} does not have any valid data between the dates {start} - {end}')
        
        # Get relevant feeds
        self.gen = tactical[self.valid_start:end]
        self.gen = self.gen[~self.gen['no_lb_period']]
        self.tactical_feed = self.gen[self.label].resample('1s').last()
        self.underlying_feed = tactical.loc[self.valid_start:end, self.underlying_name]

    def calculate_vols_and_correlations(self):
        """Calculate volatilities and correlations."""
        underlying_res = self.gen[self.underlying_name].resample('1s').last()
        tactical_log_returns = (np.log(self.tactical_feed) - np.log(self.tactical_feed.shift(1))).dropna()
        underlying_log_returns = (np.log(underlying_res) - np.log(underlying_res.shift(1))).dropna()

        self.tactical_ul_corr = np.corrcoef(tactical_log_returns, underlying_log_returns)[0,1] * 100
        self.tactical_std = tactical_log_returns.std() * np.sqrt(252 * 22 * 3600) * 100
        self.underlying_std = underlying_log_returns.std() * np.sqrt(252 * 22 * 3600) * 100

    def setup_base_plot(self, figsize=(9,8)):
        """Setup the base plot with common elements."""
        self.fig = plt.figure(figsize=figsize)
        self.grid = self.get_grid_spec()
        self.setup_axes()
        
    @abstractmethod
    def get_grid_spec(self):
        """Define the grid specification for the plot."""
        pass
        
    @abstractmethod
    def setup_axes(self):
        """Setup the axes for the plot."""
        pass
        
    @abstractmethod
    def plot_indicator(self):
        """Plot the indicator-specific elements."""
        pass

    def plot_common_elements(self, vols: bool = True, prod: Optional[pd.DataFrame] = None, 
                           prod_relative: bool = True, event: Optional[pd.Series] = None, 
                           loc: str = 'best', underlying_name_short: str = ''):
        """Plot elements common to all indicators."""
        label1 = f'Quants - Backtest'
        label2 = f'Underlying'
        
        if vols:
            self.calculate_vols_and_correlations()
            label1 += f' Vol: {self.tactical_std:.2f}%'
            label2 += f' Vol: {self.underlying_std:.2f}%'
            
        # Plot and gather lines
        line1 = self.ax0.plot(self.gen.index, self.gen[self.label], label=label1)
        line2 = self.ax0_twin.plot(self.underlying_feed.index, self.underlying_feed, 
                                  label=label2, color=DEFAULT_CMAP(2), ls=':')
        self.lines0 = line1 + line2

        # Add correlation if vols enabled
        if vols:
            label_corr = f'Correlation: {self.tactical_ul_corr:.2f}%'
            line_corr = self.ax0_twin.plot(np.nan, np.nan, '-', color='none', label=label_corr)
            self.lines0 += line_corr

        # Draw production feed if given
        if prod is not None:
            prod = prod[self.valid_start:self.end]
            line_prod = self.ax0.plot(prod.index, 
                                    prod.spot / (prod.spot.iloc[0] / self.gen[self.label].iloc[0] if prod_relative else 1),
                                    label=f'Production - Live', color=DEFAULT_CMAP(3))
            self.lines0 += line_prod

        # Draw event line if given
        if event is not None:
            event_date = pd.Timestamp(event.release_date)
            self.ax0.axvline(event_date, color='black', linestyle='--')
            if hasattr(self, 'ax1'):
                self.ax1.axvline(event_date, label=event.event_name, color='black', linestyle='--')

        # Draw legend for first subgraph
        self.ax0.legend(self.lines0, [l.get_label() for l in self.lines0], loc=loc)
        
        # Common axis labels and formatting
        self.ax0.label_outer(True)
        if hasattr(self, 'ax1'):
            self.ax1.tick_params(axis='x', labelrotation=30)
            self.ax1.set_xlabel('ts')
        else:
            self.ax0.tick_params(axis='x', labelrotation=30)
            self.ax0.set_xlabel('ts')
            
        self.ax0.set_ylabel('Tactical ($)')
        self.ax0_twin.set_ylabel(f'{self.underlying_name} ($)')
        
        # Set title
        self.fig.suptitle(
            f'{underlying_name_short if underlying_name_short else self.underlying_name} '
            f'{self.get_indicator_name()} {_renamer(self.label, underlying_name_short)} backtest '
            f'from {self.gen.index[0].round("1s")} to {self.gen.index[-1].round("1s")}',
            y=0.95
        )
        
    @abstractmethod
    def get_indicator_name(self) -> str:
        """Return the name of the indicator."""
        pass

class RSIVisualizer(IndicatorVisualizer):
    def __init__(self, tactical: pd.DataFrame, thresholds: dict[str,float], 
                 start: str | pd.Timestamp, end: str | pd.Timestamp):
        super().__init__(tactical, start, end)
        self.thresholds = thresholds
        
    def get_grid_spec(self):
        return gs.GridSpec(5,1, hspace=0.1)
        
    def setup_axes(self):
        self.ax0 = self.fig.add_subplot(self.grid[:3])
        self.ax1 = self.fig.add_subplot(self.grid[3:], sharex=self.ax0)
        self.ax0_twin = self.ax0.twinx()
        
    def plot_indicator(self):
        self.ax1.plot(self.gen.index, self.gen['rsi'], label='Quants RSI - Backtest', color='darkorange')
        self.ax1.axhline(self.thresholds['upper_threshold'], color='fuchsia', label='Upper Threshold')
        self.ax1.axhline(self.thresholds['lower_threshold'], color='aqua', label='Lower Threshold')
        self.ax1.legend(loc='lower center', bbox_to_anchor=(0.5,-0.45), ncols=4)
        self.ax1.set_ylabel('RSI')
        
    def get_indicator_name(self) -> str:
        return "RSI"

class MACOVisualizer(IndicatorVisualizer):
    def get_grid_spec(self):
        return gs.GridSpec(5,1, hspace=0.1)
        
    def setup_axes(self):
        self.ax0 = self.fig.add_subplot(self.grid[:3])
        self.ax1 = self.fig.add_subplot(self.grid[3:], sharex=self.ax0)
        self.ax0_twin = self.ax0.twinx()
        self.ax1_twin = self.ax1.twinx()
        
    def plot_indicator(self):
        line1_1 = self.ax1.plot(self.gen.index, self.gen['signal'], 
                               label='Quants MA Crossover Signal', alpha=0.6, color='darkorange')
        line2_1 = self.ax1_twin.plot(self.gen.index, self.gen['slow_ma'], 
                                    color='fuchsia', label='Slow Moving Average')
        line3_1 = self.ax1_twin.plot(self.gen.index, self.gen['fast_ma'], 
                                    color='aqua', label='Fast Moving Average')
        lines1 = line1_1 + line2_1 + line3_1
        
        self.ax1.legend(lines1, [l.get_label() for l in lines1], 
                       loc='lower center', bbox_to_anchor=(0.5,-0.45), ncols=4)
        self.ax1.set_ylabel('MACO (Signal)')
        self.ax1_twin.set_ylabel('MA ($)')
        
    def get_indicator_name(self) -> str:
        return "MACO"

class BBYVisualizer(IndicatorVisualizer):
    def get_grid_spec(self):
        return gs.GridSpec(5,1, hspace=0.1)
        
    def setup_axes(self):
        self.ax0 = self.fig.add_subplot(self.grid[:])
        self.ax0_twin = self.ax0.twinx()
        
    def plot_indicator(self):
        bands = self.tactical.loc[self.valid_start:self.end, ['upper_band', 'lower_band']].dropna()
        line_upper = self.ax0_twin.plot(bands.index, bands['upper_band'], 
                                      color='tab:red', label='Upper Band')
        line_lower = self.ax0_twin.plot(bands.index, bands['lower_band'], 
                                      color='tab:olive', label='Lower Band')
        self.lines0.extend(line_upper + line_lower)
        
    def get_indicator_name(self) -> str:
        return "BB"

def display_RSI_stats(tactical: pd.DataFrame, thresholds: dict[str,float], 
                     start: str | pd.Timestamp, end: str | pd.Timestamp, 
                     underlying_name_short: str = '', vols: bool = True, 
                     prod: Optional[pd.DataFrame] = None, prod_relative: bool = True, 
                     event: Optional[pd.Series] = None, loc: str = 'best') -> None:
    """
    Plots Tactical with underlying and RSI.
    Optionally plots production feeds and events.
    """
    visualizer = RSIVisualizer(tactical, thresholds, start, end)
    visualizer.setup_base_plot()
    visualizer.plot_indicator()
    visualizer.plot_common_elements(vols, prod, prod_relative, event, loc, underlying_name_short)
    plt.show()

def display_MACO_stats(tactical: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp, 
                      underlying_name_short: str = '', vols: bool = True, 
                      prod: Optional[pd.DataFrame] = None, prod_relative: bool = True, 
                      event: Optional[pd.Series] = None, loc: str = 'best', 
                      save: Optional[str] = None) -> None:
    """
    Plots Tactical with underlying and MACO.
    Optionally plots production feeds and events.
    """
    visualizer = MACOVisualizer(tactical, start, end)
    visualizer.setup_base_plot()
    visualizer.plot_indicator()
    visualizer.plot_common_elements(vols, prod, prod_relative, event, loc, underlying_name_short)
    if save is not None and isinstance(save, str):
        visualizer.fig.savefig(save + f'{visualizer.label}_plot_full.png')
    plt.show()

def display_BBY_stats(tactical: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp, 
                     underlying_name_short: str = '', vols: bool = True, 
                     prod: Optional[pd.DataFrame] = None, prod_relative: bool = True, 
                     event: Optional[pd.Series] = None, loc: str = 'best') -> None:
    """
    Plots Tactical with underlying and Bollinger Bands.
    Optionally plots production feeds and events.
    """
    visualizer = BBYVisualizer(tactical, start, end)
    visualizer.setup_base_plot(figsize=(9,5))
    visualizer.plot_indicator()
    visualizer.plot_common_elements(vols, prod, prod_relative, event, loc, underlying_name_short)
    plt.show()