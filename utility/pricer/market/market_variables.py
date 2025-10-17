import numpy as np

from .interestrates.yieldcurve import YieldCurve
from .interestrates.yieldcurve_flat import YieldCurveFlat
from .volsurface.volsurface import VolSurface

from typing import Union


VALID_MARKETS = ["forex",
                 "equities",
                 "commodities",
                 "synthetic",
                 "derived",
                 ]


class MarketSnapshot():
    """
    A class containing a snapshot of market variables required to price an option

    Attributes
    ----------
    market : str
        The market type for reference
    interest_rates : YieldCurve
        The interest rates in the market
    dividend_rates : YieldCurve
        The dividend rates in the market
    volsurface :  VolSurface
        The volatility surface in the market
    """

    def __init__(self,
                 market: str,
                 interest_rates: Union[YieldCurve,
                                       float],
                 dividend_rates: Union[YieldCurve,
                                       float],
                 volsurface: Union[VolSurface,
                                   float]) -> None:
        if market in VALID_MARKETS:
            self.market = market
        else:
            raise ValueError(
                f"Provided market is invalid. Try one of {VALID_MARKETS}")

        # Instantiate market data objects
        if isinstance(interest_rates, YieldCurve):
            self.interest_rates = interest_rates
        elif isinstance(interest_rates, float):
            self.interest_rates = YieldCurveFlat(
                np.array([[None, interest_rates]]))
        else:
            raise ValueError(
                "Invalid type for interest rates. Try one of [YieldCurveFlat, YieldCurveDiscrete, YieldCurveLinear, float]")

        if isinstance(dividend_rates, YieldCurve):
            self.dividend_rates = dividend_rates
        elif isinstance(dividend_rates, float):
            self.dividend_rates = YieldCurveFlat(
                np.array([[None, dividend_rates]]))
        else:
            raise ValueError(
                "Invalid type for dividend rates. Try one of [YieldCurveFlat, YieldCurveDiscrete, YieldCurveLinear, float]")

        if isinstance(volsurface, VolSurface):
            self.volsurface = volsurface
        elif isinstance(volsurface, float):
            self.volsurface = VolSurface(volsurface)
        else:
            raise ValueError(
                "Invalid type for the volatility surface. Try one of [VolSurface, float]")

    def get_interest_rate(self, T: float) -> float:
        """
        Retrieve interpolated interest rates

        Parameters
        ----------
        T : float
            The time to maturity of the contract

        Returns
        -------
        float
            The interpolated interest rate
        """
        return self.interest_rates.get_rate(T)

    def get_dividend_rate(self, T: float) -> float:
        """
        Retrieve interpolated dividend rates

        Parameters
        ----------
        T : float
            The time to maturity of the contract

        Returns
        -------
        float
            The interpolated dividend rate
        """
        return self.dividend_rates.get_rate(T)

    def get_volatility(self, T: float, *args: float) -> np.ndarray:
        """
        Retrieve interpolated volatilities

        Parameters
        ----------
        T : float
            The time to maturity of the contract
        *args
            The strike prices

        Returns
        -------
        numpy.ndarray
            The interpolated volatilities
        """
        return self.volsurface.get_volatility(T, *args)

    def get_interest_rate_curve(self) -> YieldCurve:
        """
        Retrieve the interest rate curve

        Returns
        -------
        YieldCurve
            The interest rate curve
        """
        return self.interest_rates

    def get_dividend_rate_curve(self):
        """
        Retrieve the dividend rate curve

        Returns
        -------
        YieldCurve
            The dividend rate curve
        """
        return self.dividend_rates

    def get_volsurface(self) -> VolSurface:
        """
        Retrieve the volatility surface

        Returns
        -------
        VolSurface
            The volatility surface
        """
        return self.volsurface
