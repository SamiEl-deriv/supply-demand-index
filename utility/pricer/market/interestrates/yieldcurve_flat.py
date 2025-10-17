import numpy as np
from .yieldcurve import YieldCurve
import warnings


class YieldCurveFlat(YieldCurve):
    """
    A flat yield curve object

    Attributes
    ----------
    market_data : numpy.ndarray
        An array of [time_in_years, rate]-pairs.
        Only the first element is considered in a flat curve
    rate : float
        The single rate to use
    """

    def __init__(self, market_data: np.ndarray) -> None:
        """
        Parameters
        ----------
        market_data : numpy.ndarray
            An array of [time_in_years, rate]-pairs.
            Only the first element is considered in a flat curve
        """
        super().__init__(market_data)

        if self.market_data.shape[0] != 1:
            warnings.warn(
                "Market data has more than 1 row. Only the first row will be considered")

        self.rate = self.market_data[0, 1]

    def get_rate(self, t) -> float:
        """
        Returns flat rate

        Returns
        -------
        float
            Flat rate
        """
        return self.rate
