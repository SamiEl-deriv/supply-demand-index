import numpy as np
from .volsmile import VolSmile


class VolSmileFlat(VolSmile):
    """
    A class that returns a constant volatility

    Attributes
    ----------
    vol : float
        Flat (constant) volatility
    """

    def __init__(self, vol=0.1) -> None:
        """
        Parameters
        ----------
        vol : float
            Flat volatility, default 0.1
        """

        self.vol = vol

    def __call__(self, *args) -> float:
        """
        Returns
        -------
        float
            Flat volatility
        """

        return self.get_vol(args)

    def __str__(self) -> str:
        """
        Prints volatility smile details

        Returns
        -------
        str
            volatility smile details - volatility
        """

        return f"Flat Volatility Smile\nVolatility: {self.vol}"

    def get_vol(self, *args) -> float:
        """
        Returns
        -------
        float
            Flat volatility
        """

        # Hackish solution to preserve dimension of returned value
        if isinstance(
                args, tuple) or not isinstance(
                args, (float, np.ndarray)):
            args = np.array(args)
            args = args.flatten()
        return (args - args) + self.vol
