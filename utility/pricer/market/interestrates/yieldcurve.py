from abc import ABC, abstractmethod
import numpy as np
import scipy.integrate as integrate


class YieldCurve(ABC):
    """
    The abstract yield curve class

    Attributes
    ----------
    market_data : numpy.ndarray
        An array of [time_in_years, rate]-pairs
    """

    def __init__(self, market_data: np.ndarray) -> None:
        """
        Parameters
        ----------
        market_data : numpy.ndarray
            An array of [time_in_years, rate]-pairs
        """
        market_data_array = np.array(market_data)

        if market_data_array.shape[1] != 2:
            raise ValueError(
                "Market data array provided is not properly formatted, requires 2 columns:\n"
                "  array([[t, rate] for t in t_range])")
        self.market_data = market_data_array

    @abstractmethod
    def get_rate(self, t):
        """
        Abstract method to return interest rates based on the curve
        """
        return NotImplementedError("Implementation Required")

    def integrate(self, T: float):
        """
        Retrieve the integrated rate over a period t

        Parameters
        ----------
        T: float
            The period through which the rate is integrated.
        """
        start_time = self.market_data[0, 0]
        end_time = start_time + T
        integrated_rate, _ = integrate.quad(lambda t: self.get_rate(t),
                                            start_time, end_time)
        return round(integrated_rate, 10)
