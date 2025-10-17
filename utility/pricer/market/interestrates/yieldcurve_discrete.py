from .yieldcurve import YieldCurve
import numpy as np
import warnings


class YieldCurveDiscrete(YieldCurve):
    """
    A piecewise-constant interpolated yield curve object
    Deriv uses this for dividend rates

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
        Returns rate corresponding the time-in-years entry closest to the given argument

        Parameters
        ----------
        t : float
            The queried time-in-years

        Returns
        -------
        float
            The closest interest rate
        """
        t_range = self.market_data[:, 0]

        closest_index = np.argmin(t_range - t)

        return self.market_data[closest_index, 1]
