from stochastic_process_base import StochasticProcess

from typing import Type
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def run_plot(self) -> None:
    plt.ion()
    fig, ax = plt.subplots()
    try:
        while True:
            self.update()
            print(self._StochasticProcess__last_value, self._StochasticProcess__bid[-1], self._StochasticProcess__ask[-1])
            x_data = self.index
            y_data = np.arange(len(self.index))
            ax.plot(y_data, x_data)
            ax.plot(y_data, self._StochasticProcess__bid)
            ax.plot(y_data, self._StochasticProcess__ask)
            plt.draw()
            plt.pause(0.01)
            plt.cla()
            self.sleep()
    except KeyboardInterrupt:
        print("Execution stopped manually.")


def run_plot_spread(self) -> None:
    plt.ion()
    fig, ax = plt.subplots()
    try:
        while True:
            self.update()
            print("spread is :", self._StochasticProcess__bid[-1]- self._StochasticProcess__ask[-1])
            y_data = np.arange(len(self.index))
            ax.plot(y_data, self._StochasticProcess__spread)
            plt.draw()
            plt.pause(0.01)
            plt.cla()
            self.sleep()
    except KeyboardInterrupt:
        print("Execution stopped manually.")


def get_stats(self) -> list:
    vol = np.std(np.diff(self.index))
    trend = np.mean(np.diff(self.index))
    skew = stats.skew(np.diff(self.index))
    kurtosis = stats.kurtosis(np.diff(self.index))
    print(' vol is {:.3f},\n trend is {:.3f},\n skew is {:.3f},\n kurtosis is {:.3f}'.format(vol,trend,skew,kurtosis))
    return [vol, trend, skew, kurtosis]


# further tools will be here

# use this function to add prvious funcitons as methods to a StochasticProcess object
def mount_tool_kit(instance):
    instance.run_plot = run_plot.__get__(instance)
    instance.run_plot_spread = run_plot_spread.__get__(instance)
    instance.get_stats = get_stats.__get__(instance)

