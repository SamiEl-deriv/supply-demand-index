import numpy as np
from scipy.interpolate import interp1d
from .volsmile import VolSmile


class VolSmileSpline(VolSmile):
    """
    A class that interpolates/extrapolates volatility using a polynomial
    Kinds used: Quadratic, Cubic spline
    """

    def __init__(self, type, market_data) -> None:
        """
        Constructs the spline used for interpolation
        Quadratic interpolation requires 3 points
        Cubic spline interpolation requires 4 points

        Parameters
        ----------
        type : str
            A string denoting the kind of interpolation to use (Currently only "cubic" | "quadratic")
        market_data : dict(float : float)
            A dictionary containing strike/delta/moneyness-volatility pairs.
        """

        self.market_data = market_data
        x = list(market_data.keys())
        y = list(market_data.values())
        self.type = type

        # Catch exceptions
        if len(x) != len(y):
            raise ValueError("Both arrays must have the same length")
        if type not in ["quadratic", "cubic"]:
            raise ValueError("Type must be either quadratic or cubic")
        if len(x) < 3 and type == "quadratic":
            raise ValueError(
                "Quadratic interpolation requires at least 3 points")
        elif len(x) < 4 and type == "cubic":
            raise ValueError("Cubic interpolation requires at least 4 points")

        self.spline = interp1d(x, y, kind=self.type, fill_value='extrapolate')

    def __call__(self, *args) -> float:
        """
        Constructs the spline used for interpolation

        Parameters
        ----------
        y : arr(float)
            float array of y-values
        """

        return self.get_vol(args)

    def __str__(self) -> str:
        """
        Prints volatility market_data details

        Returns
        -------
        str
            volatility market_data details - quadratic/cubic, market_data
        """

        return f"{self.type.capitalize()} Volatility Curve\nmarket_data: {self.market_data}"

    def get_vol(self, *args) -> float:
        """
        Retrieves an interpolated volatility

        Parameters
        ----------
        y : arr(float)
            float array of y-values
        """

        return self.spline(args)

    def get_market_data(self) -> dict:
        """
        Returns the market_data as a dict

        Returns
        -------
        dict
            A dict with keys equal to the x-array and values euqal to the y-array
        """

        return self.market_data
