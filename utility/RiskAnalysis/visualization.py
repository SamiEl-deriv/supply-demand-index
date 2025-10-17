import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


class Visualizer:
    """Class for visualizing financial analysis results."""

    def __init__(self):
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False

    def plot_moments(self, moments, show_plot=True):
        """
        Plot rolling moments.

        Args:
            moments (dict): Dictionary containing moment series.
            show_plot (bool): Whether to display the plot immediately.

        Returns:
            tuple: (figure, axes) matplotlib objects
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot mean and volatility
        axes[0].plot(moments['mean'].index, moments['mean'], 
                    label='Mean', color='blue')
        axes[0].plot(moments['volatility'].index, moments['volatility'], 
                    label='Volatility', color='red')
        axes[0].set_title('Rolling Mean and Volatility')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot skewness
        axes[1].plot(moments['skewness'].index, moments['skewness'], 
                    label='Skewness', color='green')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1].set_title('Rolling Skewness')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot kurtosis
        axes[2].plot(moments['kurtosis'].index, moments['kurtosis'], 
                    label='Kurtosis', color='purple')
        axes[2].axhline(y=3, color='black', linestyle='--', alpha=0.3)
        axes[2].set_title('Rolling Kurtosis')
        axes[2].legend()
        axes[2].grid(True)
        
        # Format x-axis
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        
        plt.tight_layout()
        if show_plot:
            plt.show()
        
        return fig, axes

    def plot_drawdown(self, returns, show_plot=True):
        """
        Visualize cumulative returns and drawdown.

        Args:
            returns (pd.Series): Series of returns.
            show_plot (bool): Whether to display the plot immediately.

        Returns:
            tuple: (figure, axes) matplotlib objects
        """
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown_series = (cumulative_returns - peak) / peak

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot Cumulative Returns
        axes[0].plot(returns.index, cumulative_returns, color='blue', label='Cumulative Returns')
        axes[0].set_title('Cumulative Returns')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot Drawdown
        axes[1].fill_between(returns.index, drawdown_series, 0, color='red', alpha=0.5, label='Drawdown')
        axes[1].set_title('Drawdown')
        axes[1].legend()
        axes[1].grid(True)

        # Format x-axis
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

        plt.xlabel('Time')
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        
        return fig, axes

    def plot_autocorrelation(self, acf, pacf, conf_level, show_plot=True):
        """
        Plot autocorrelation and partial autocorrelation functions.

        Args:
            acf (pd.Series): Autocorrelation values.
            pacf (pd.Series): Partial autocorrelation values.
            conf_level (float): Confidence level for significance bands.
            show_plot (bool): Whether to display the plot immediately.

        Returns:
            tuple: (figure, axes) matplotlib objects
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot ACF
        axes[0].bar(range(len(acf)), acf, alpha=0.5, color='blue')
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0].axhline(y=conf_level, color='red', linestyle='--', alpha=0.5)
        axes[0].axhline(y=-conf_level, color='red', linestyle='--', alpha=0.5)
        axes[0].set_title('Autocorrelation Function')
        axes[0].set_xlabel('Lag')
        axes[0].grid(True)
        
        # Plot PACF
        axes[1].bar(range(len(pacf)), pacf, alpha=0.5, color='blue')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].axhline(y=conf_level, color='red', linestyle='--', alpha=0.5)
        axes[1].axhline(y=-conf_level, color='red', linestyle='--', alpha=0.5)
        axes[1].set_title('Partial Autocorrelation Function')
        axes[1].set_xlabel('Lag')
        axes[1].grid(True)
        
        plt.tight_layout()
        if show_plot:
            plt.show()
        
        return fig, axes

    def plot_rolling_metrics(self, dates, rolling_beta, realized_vol, show_plot=True):
        """
        Plot rolling beta and realized volatility.

        Args:
            dates (pd.DatetimeIndex): Dates for x-axis.
            rolling_beta (pd.Series): Rolling beta values.
            realized_vol (pd.Series): Realized volatility values.
            show_plot (bool): Whether to display the plot immediately.

        Returns:
            tuple: (figure, axes) matplotlib objects
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot rolling beta
        axes[0].plot(dates[60:], rolling_beta[60:], color='blue', label='Rolling Beta')
        axes[0].set_title('Rolling Beta')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot realized volatility
        axes[1].plot(dates[21:], realized_vol[21:], color='red', label='Realized Volatility')
        axes[1].set_title('Realized Volatility')
        axes[1].legend()
        axes[1].grid(True)
        
        # Format x-axis
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        
        plt.tight_layout()
        if show_plot:
            plt.show()
        
        return fig, axes
