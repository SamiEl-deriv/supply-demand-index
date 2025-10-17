from .yieldcurve import YieldCurve
from ..utils import linear_interpolate
import numpy as np
import warnings


class YieldCurveLinear(YieldCurve):
    """
    A piecewise-linearly interpolated yield curve object

    Attributes
    ----------
    market_data : numpy.ndarray
        An array of [time_in_years, rate]-pairs
    """

    def __init__(self, market_data: np.ndarray) -> None:
        super().__init__(market_data)

        if self.market_data.shape[0] == 1:
            warnings.warn(
                "market_data has only one row. Consider using YieldCurveFlat instead")

    def get_rate(self, t: float) -> float:
        """
        Returns a linearly interpolated rate given the time-in-years

        Parameters
        ----------
        t : float
            The queried time-in-years

        Returns
        -------
        float
            The interpolated interest rate
        """
        t_range = self.market_data[:, 0]

        t_left = np.searchsorted(t_range, t) - 1

        line_segment = self.market_data[t_left: t_left + 2, :]

        return np.interp(t, line_segment[:, 0], line_segment[:, 1])
